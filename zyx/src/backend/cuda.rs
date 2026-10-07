// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#![allow(unused)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::needless_continue)]
#![allow(clippy::unnecessary_semicolon)]
#![allow(clippy::manual_assert)]
#![allow(clippy::get_first)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::useless_format)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::similar_names)]
#![allow(clippy::len_zero)]
#![allow(clippy::question_mark)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_string_new)]

// ── Global state ──────────────────────────────────────────────────────────────

static CUDA_POOLS: OnceLock<Vec<Mutex<CUDAMemoryPool>>> = OnceLock::new();
static CUDA_DEVICES: OnceLock<Vec<Mutex<CUDADevice>>> = OnceLock::new();

const VEC_COMPONENTS: [&str; 16] = [
    "x", "y", "z", "w", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sa", "sb",
];

// cuDNN v9 graph API constants (from cudnn_graph_v9.h). The library is dlopen'd
// at runtime like libcuda; these are hardcoded so no C headers are needed.
const CUDNN_STATUS_SUCCESS: c_int = 0;

const CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: c_int = 15;
const CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: c_int = 16;
const CUDNN_BACKEND_TENSOR_DESCRIPTOR: c_int = 17;
const CUDNN_BACKEND_MATMUL_DESCRIPTOR: c_int = 18;
const CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: c_int = 19;
const CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: c_int = 4;
const CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: c_int = 5;

const CUDNN_ATTR_TENSOR_DATA_TYPE: c_int = 901;
const CUDNN_ATTR_TENSOR_DIMENSIONS: c_int = 902;
const CUDNN_ATTR_TENSOR_STRIDES: c_int = 903;
const CUDNN_ATTR_TENSOR_UNIQUE_ID: c_int = 906;
const CUDNN_ATTR_TENSOR_IS_VIRTUAL: c_int = 907;

const CUDNN_ATTR_MATMUL_COMP_TYPE: c_int = 1500;

const CUDNN_ATTR_OPERATION_MATMUL_ADESC: c_int = 1520;
const CUDNN_ATTR_OPERATION_MATMUL_BDESC: c_int = 1521;
const CUDNN_ATTR_OPERATION_MATMUL_CDESC: c_int = 1522;
const CUDNN_ATTR_OPERATION_MATMUL_DESC: c_int = 1523;

const CUDNN_ATTR_OPERATIONGRAPH_OPS: c_int = 801;

const CUDNN_ATTR_ENGINEHEUR_MODE: c_int = 200;
const CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH: c_int = 201;
const CUDNN_ATTR_ENGINEHEUR_RESULTS: c_int = 202;

const CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG: c_int = 401;
const CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: c_int = 402;

const CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS: c_int = 1000;
const CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS: c_int = 1001;
const CUDNN_ATTR_VARIANT_PACK_WORKSPACE: c_int = 1003;

const CUDNN_TYPE_DATA_TYPE: c_int = 1;
const CUDNN_TYPE_BOOLEAN: c_int = 2;
const CUDNN_TYPE_INT64: c_int = 3;
const CUDNN_TYPE_VOID_PTR: c_int = 6;
const CUDNN_TYPE_HEUR_MODE: c_int = 8;
const CUDNN_TYPE_BACKEND_DESCRIPTOR: c_int = 15;

const CUDNN_DATA_FLOAT: c_int = 0;
const CUDNN_DATA_HALF: c_int = 2;
const CUDNN_DATA_BFLOAT16: c_int = 9;

const CUDNN_HEUR_MODE_INSTANT: c_int = 0;

type cudnnHandle_t = *mut cudnnContext;
type cudnnBackendDescriptor_t = *mut cudnnBackend;
type cudnnDataType_t = c_int;
type cudnnStatus_t = c_int;

#[repr(C)]
#[derive(Debug)]
struct cudnnContext {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug)]
struct cudnnBackend {
    _unused: [u8; 0],
}

use std::{
    collections::BTreeSet,
    ffi::{CString, c_char, c_int, c_uint, c_void},
    hash::BuildHasherDefault,
    path::PathBuf,
    ptr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
};

use crate::hashers::FHasher;
use libloading::Library;
use nanoserde::DeJson;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::Graph,
    kernel::{GPUOp, Kernel, MMADType, MMADims, Op, OpId, PTXOp, ParamKind, RangeKind},
    shape::Dim,
    slab::{Slab, SlabId},
};

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

use super::{
    ChunkId, Cmd, DTypeCapability, Dev, DeviceInfo, DeviceProgramId, GwsDim, LaunchArg, Placement, PlanDim, Pool, ProgramId,
    Shard, gws_from_kernel,
};

/// CUDA configuration
#[allow(clippy::question_mark)]
#[derive(Debug, Default, DeJson)]
#[nserde(default)]
pub struct CUDAConfig {
    /// If set to None, then it will automatically use all CUDA devices,
    /// otherwise it uses only selected devices
    device_ids: Option<Vec<i32>>,
    /// Whether to use cuDNN for AOT matmul kernels. Defaults to true.
    cudnn: bool,
    /// Number of hardware queues (CUDA streams) per device used by the
    /// batched-submission worker. Defaults to 12 when unset.
    queues: Option<usize>,
}

#[derive(Debug)]
pub struct CUDAMemoryPool {
    tx: Sender<CUDACommand>,
    free_bytes: Arc<AtomicU64>,
    /// CUDA device handle owned by this pool's worker thread.
    pub(super) device: CUdevice,
    /// Driver ordinal (nvidia-smi id) of this pool's GPU.
    pub(super) dev_ordinal: i32,
    /// Resolved driver symbols, copied per pool so device init never
    /// needs a process-global driver handle.
    pub(super) fns: CudaFns,
    pub(super) cudnn_available: bool,
    /// Keeps libcuda loaded for the worker thread's symbols.
    _lib: Arc<Library>,
}

#[derive(Debug)]
pub(super) struct CUDABuffer {
    pub ptr: u64,
    pub bytes: Dim,
}

#[derive(Debug)]
pub struct CUDADevice {
    tx: Sender<CUDACommand>,
    device: CUdevice,
    /// Real CUDA driver ordinal (nvidia-smi id), set at init. Not the pool ordinal.
    pub(crate) dev_id: u32,
    memory_pool: Pool,
    dev_info: Arc<DeviceInfo>,
    pub compute_capability: [c_int; 2],
    cudnn_available: bool,
}

/// Partition-held handle of a worker-captured graph executable. Dropping
/// it destroys the graph on the worker (fire-and-forget — drop never
/// blocks or fails loudly).
#[derive(Debug)]
pub(super) struct CudaGraph {
    id: CudaGraphId,
    tx: Sender<CUDACommand>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let _ = self.tx.send(CUDACommand::DestroyGraph { id: self.id });
    }
}

#[derive(Debug)]
pub(super) enum CUDAProgram {
    Module {
        module: CUmodule,
        function: CUfunction,
        lws: [u32; 3],
        gws: Vec<GwsDim>,
        /// Per-`Param` kinds in head order, used at submission to split launch
        /// args into reads (`Global`) and writes (`GlobalMut`).
        params: Vec<ParamKind>,
    },
    /// A compiled cuDNN graph execution plan. Workspace is a raw device pointer
    /// allocated alongside the plan (cuDNN owns it, not the memory pool).
    Cudnn { plan: CudnnPlan },
}

/// A cuDNN v9 graph execution plan plus the metadata needed to launch it:
/// the ordered tensor UIDs for the variant pack and the workspace pointer.
#[derive(Debug)]
pub(super) struct CudnnPlan {
    plan: cudnnBackendDescriptor_t,
    /// Tensor UIDs in launch-arg order: inputs then outputs.
    arg_uids: Vec<i64>,
    workspace: u64,
    workspace_bytes: Dim,
    /// All intermediate descriptors created while building the plan, in
    /// creation order. Destroyed in reverse at release.
    descrs: Vec<cudnnBackendDescriptor_t>,
}

/// dlopen'd cuDNN library: the v9 graph API functions plus the handle/descriptor
/// create/destroy. Kept in a struct so the library stays loaded for the worker
/// thread and symbols are resolved once at init.
struct CudnnLib {
    #[allow(dead_code)]
    _lib: Library,
    create: unsafe extern "C" fn(*mut cudnnHandle_t) -> cudnnStatus_t,
    destroy: unsafe extern "C" fn(cudnnHandle_t) -> cudnnStatus_t,
    backend_create_descriptor: unsafe extern "C" fn(c_int, *mut cudnnBackendDescriptor_t) -> cudnnStatus_t,
    backend_destroy_descriptor: unsafe extern "C" fn(cudnnBackendDescriptor_t) -> cudnnStatus_t,
    backend_finalize: unsafe extern "C" fn(cudnnBackendDescriptor_t) -> cudnnStatus_t,
    backend_set_attribute: unsafe extern "C" fn(cudnnBackendDescriptor_t, c_int, c_int, i64, *const c_void) -> cudnnStatus_t,
    backend_get_attribute:
        unsafe extern "C" fn(cudnnBackendDescriptor_t, c_int, c_int, i64, *mut i64, *mut c_void) -> cudnnStatus_t,
    backend_execute: unsafe extern "C" fn(cudnnHandle_t, cudnnBackendDescriptor_t, cudnnBackendDescriptor_t) -> cudnnStatus_t,
}

/// A generic description of a cuDNN graph subgraph, mirroring the cuDNN v9
/// graph API. Sent to the worker thread for JIT compilation at match time.
#[derive(Debug)]
pub(super) struct CudnnGraph {
    tensors: Vec<CudnnTensor>,
    ops: Vec<CudnnOp>,
    /// Non-virtual tensor UIDs in launch-arg order (inputs then outputs).
    arg_uids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub(super) struct CudnnTensor {
    uid: i64,
    shape: Vec<Dim>,
    dtype: DType,
    /// Virtual tensors are intermediates; only non-virtual tensors appear in
    /// the launch arg list.
    is_virtual: bool,
}

#[derive(Debug, Clone)]
pub(super) enum CudnnOp {
    Matmul { a: i64, b: i64, c: i64, compute_dtype: DType },
}

#[derive(Debug)]
pub(super) struct CUDAStream {
    stream: CUstream,
}

/// Worker-side id of a captured CUDA graph executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CudaGraphId(u32);

impl From<usize> for CudaGraphId {
    fn from(value: usize) -> Self {
        CudaGraphId(u32::try_from(value).unwrap())
    }
}

impl From<CudaGraphId> for usize {
    fn from(value: CudaGraphId) -> Self {
        value.0 as usize
    }
}

impl SlabId for CudaGraphId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// A captured partition executable: the instantiated graph plus the
/// launch-arg slots and their device pointers at last capture. The
/// slots fix the comparison order; the pointers detect the steady
/// state (identical pointers + vars: relaunch with no work at all).
pub(super) struct CapturedGraph {
    exec: CUgraphExec,
    /// Launch-arg slots per kernel launch, in capture order.
    node_args: Vec<Vec<OpId>>,
    /// Device pointers of buffer slots at last capture, flat in the
    /// same order as `node_args` (one entry per buffer slot).
    ptrs: Vec<u64>,
    /// Scalar values at last capture.
    vars: Vec<(OpId, Constant)>,
}

/// Worker reply to `Replay`: freshly allocated chunks plus the captured
/// graph id (`None` when nothing was captured: dry run or direct submit).
/// Fresh triples carry the region length (evaluated bytes, offset 0) so the
/// caller names the live region on the (possibly over-allocated) chunk.
pub(super) struct ReplayResult {
    pub fresh: Vec<(OpId, ChunkId, usize)>,
    pub graph: Option<CudaGraphId>,
}

enum CUDACommand {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<ChunkId, BackendError>>,
    },
    Dispose,
    Release {
        buffer_id: ChunkId,
    },
    /// Try to re-claim every id in the set from this pool's free list
    /// (all-or-nothing): `Ok(true)` = stable addresses, `Ok(false)` =
    /// slots changed, invalidate the graph and retrace.
    TryReuse {
        buffer_ids: Set<ChunkId>,
        reply: Sender<Result<bool, BackendError>>,
    },
    /// Blocking read-back: the reply is sent after the data arrived in
    /// host memory. This is a sync point — all pending work is submitted
    /// and the device is drained first.
    PoolToHost {
        src: ChunkId,
        dst: *mut u8,
        bytes: Dim,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Export the device pointer, context and in-flight barrier events of a
    /// buffer so another device's worker can copy from it (peer-copy source side).
    Compile {
        lws: [u32; 3],
        gws: Vec<GwsDim>,
        /// Per-`Param` kinds in head order, used at submission to split launch
        /// args into reads (`Global`) and writes (`GlobalMut`).
        params: Vec<ParamKind>,
        name: Box<str>,
        ptx: Vec<u8>,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    /// JIT-compiles a cuDNN graph execution plan for the given subgraph. Shapes
    /// and dtypes are fixed at match time. The command mirrors the cuDNN graph
    /// API generically (tensors + ops), so it works for matmul, matmul+relu,
    /// softmax, conv, etc. — only the matmul builder is implemented so far.
    CompileCudnn {
        graph: CudnnGraph,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    /// Timed launch for autotune: flush + drain first (uncontended timing),
    /// then launch solo on one stream, bracket with timing events, reply nanos.
    LaunchTimed {
        program_id: DeviceProgramId,
        args: Vec<LaunchArg>,
        reply: Sender<Result<u64, BackendError>>,
    },
    /// Execute a preplanned partition: allocate unbound defs from their
    /// specs, capture to a CUDA graph on first sight (or patch + relaunch
    /// the `graph` executable), launch, reply fresh chunks + graph id.
    /// Partitions holding cuDNN programs are submitted directly and never
    /// captured. Dynamic-shape repeats (vars mismatch) recapture in place
    /// under the same id.
    Replay {
        cmds: Vec<Cmd>,
        /// Bound slots (inputs + kept outputs): queue-local slot to resident chunk.
        bound: Vec<(OpId, ChunkId, usize)>,
        /// Scalar values for variable slots + dynamic dims.
        vars: Vec<(OpId, Constant)>,
        /// Device-side reuse hint: a live captured-graph id, if any.
        graph: Option<CudaGraphId>,
        reply: Sender<Result<ReplayResult, BackendError>>,
    },
    /// Blocking host-to-device upload for Copy partitions into CUDA.
    /// The host pointer stays valid for the roundtrip (caller holds the
    /// host pool lock across send + recv).
    CopyHtoD {
        src: *const u8,
        dst: ChunkId,
        bytes: Dim,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Blocking upload straight from a file mapping: registers the mapped
    /// range for DMA, uploads, unregisters — all on the worker thread, which
    /// owns the CUDA context (caller threads have none). No staging buffer,
    /// no read call.
    CopyDiskToD {
        ptr: *const u8,
        extent: Dim,
        bytes: Dim,
        dst: ChunkId,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Blocking same-device device-to-device copy (single pool).
    CopyDtoD {
        src: ChunkId,
        dst: ChunkId,
        bytes: Dim,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Destroy a captured graph exec (from `CudaGraph::drop`): tolerant —
    /// an unknown id is already gone.
    DestroyGraph {
        id: CudaGraphId,
    },
    ReleaseProgram {
        program_id: DeviceProgramId,
    },
}

unsafe impl Send for CUDACommand {}

/// Resolved CUDA driver symbols (`Copy`, so each pool owns its copy and no
/// process-global driver handle is needed). The `Library` itself lives in an
/// `Arc` on each pool + worker thread, OpenCL-style, keeping symbols valid.
#[derive(Debug, Clone, Copy)]
pub(super) struct CudaFns {
    cuInit: unsafe extern "C" fn(c_uint) -> CUDAStatus,
    cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUDAStatus,
    cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUDAStatus,
    cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUDAStatus,
    cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUDAStatus,
    cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUDAStatus,
    cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUDAStatus,
    cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUDAStatus,
    cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUDAStatus,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus,
    cuMemGetInfo: unsafe extern "C" fn(*mut usize, *mut usize) -> CUDAStatus,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUDAStatus,
    cuMemcpyHtoDAsync: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUDAStatus,
    cuMemcpyDtoHAsync: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUDAStatus,
    cuModuleLoadDataEx:
        unsafe extern "C" fn(*mut CUmodule, *const c_void, c_uint, *mut CUjit_option, *mut *mut c_void) -> CUDAStatus,
    cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUDAStatus,
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
    cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUDAStatus,
    cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus,
    cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUDAStatus,
    cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUDAStatus,
    cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUDAStatus,
    cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUDAStatus,
    cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUDAStatus,
    /// Non-blocking poll: CUDA_SUCCESS if complete, CUDA_ERROR_NOT_READY if not.
    cuEventQuery: unsafe extern "C" fn(CUevent) -> CUDAStatus,
    /// Elapsed time between two completed events in milliseconds (needs
    /// timing-enabled events, i.e. NOT created with CU_EVENT_DISABLE_TIMING).
    cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUDAStatus,
    cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUDAStatus,
    cuCtxEnablePeerAccess: unsafe extern "C" fn(CUcontext, c_uint) -> CUDAStatus,
    cuMemcpyPeerAsync: unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize, CUstream) -> CUDAStatus,
    cuStreamBeginCapture: unsafe extern "C" fn(CUstream, CUstreamCaptureMode) -> CUDAStatus,
    cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUDAStatus,
    cuGraphInstantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, *mut c_void, *mut c_char, usize) -> CUDAStatus,
    cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUDAStatus,
    cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUDAStatus,
    cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUDAStatus,
    cuGraphExecUpdate: unsafe extern "C" fn(CUgraphExec, CUgraph, *mut CUgraphExecUpdateResultInfo) -> CUDAStatus,
    cuMemcpyDtoDAsync: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUDAStatus,
}

/// Single entry point: builds pools + devices together in one enumeration,
/// publishes both tables, returns both. No init locks — `OnceLock::set`
/// publishes exactly once; losers' workers exit when their channels drop.
fn backend() -> Result<(&'static Vec<Mutex<CUDAMemoryPool>>, &'static Vec<Mutex<CUDADevice>>), BackendError> {
    if let Some(pools) = CUDA_POOLS.get()
        && let Some(devs) = CUDA_DEVICES.get()
    {
        return Ok((pools, devs));
    }
    let (pools, devs) = initialize_backend()?;
    let _ = CUDA_POOLS.set(pools);
    let _ = CUDA_DEVICES.set(devs);
    match (CUDA_POOLS.get(), CUDA_DEVICES.get()) {
        (Some(pools), Some(devs)) => Ok((pools, devs)),
        _ => Err(BackendError { status: ErrorStatus::Initialization, context: "CUDA init failed".into() }),
    }
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<CUDAMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::Cuda({id}) is not available").into() }
}

fn load_driver() -> Result<(Arc<Library>, Option<Arc<CudnnLib>>, Vec<i32>, CudaFns), BackendError> {
    let debug_dev = super::debug_backends();
    let config = super::config();
    if let Some(device_ids) = &config.cuda.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[cuda] configured out");
        }
        return Err(BackendError { status: ErrorStatus::Initialization, context: "[cuda] configured out".into() });
    }

    let cuda_paths = [
        "libcuda.so.1",
        "libcuda.so",
        "/lib64/libcuda.so",
        "/lib/libcuda.so",
        "/usr/lib64/libcuda.so",
        "/usr/lib/libcuda.so",
        "/lib/x86_64-linux-gnu/libcuda.so",
        "/lib64/x86_64-linux-gnu/libcuda.so",
    ];
    let cuda = cuda_paths.into_iter().find_map(|path| unsafe { Library::new(path) }.ok());

    let Some(cuda) = cuda else {
        if debug_dev {
            println!("[cuda] libcuda.so not found");
        }
        return Err(BackendError { status: ErrorStatus::DyLibNotFound, context: "[cuda] libcuda.so not found.".into() });
    };

    // Load cuDNN for AOT matmul kernels (optional). Kept alive for the worker
    // threads via an Arc; without it the CUDA backend still works normally.
    let cudnn = if config.cuda.cudnn { load_cudnn() } else { None };
    if debug_dev && cudnn.is_some() {
        println!("[cuda] cuDNN graph API loaded");
    } else if debug_dev && !config.cuda.cudnn {
        println!("[cuda] cuDNN disabled by config");
    }

    let cuInit: unsafe extern "C" fn(c_uint) -> CUDAStatus = *unsafe { cuda.get(b"cuInit\0") }?;
    let cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUDAStatus = *unsafe { cuda.get(b"cuDriverGetVersion\0") }?;
    let cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceGetCount\0") }?;
    let cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceGet\0") }?;
    let cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetName\0") }?;
    let cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceComputeCapability\0") }?;
    let cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceTotalMem_v2\0") }?;
    let cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetAttribute\0") }?;
    let cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuCtxCreate_v2\0") }?;
    //let cuMemAllocAsync = *unsafe { cuda.get(b"cuMemAllocAsync\0") }?;
    let cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus = *unsafe { cuda.get(b"cuMemAlloc_v2\0") }?;
    let cuMemGetInfo: unsafe extern "C" fn(*mut usize, *mut usize) -> CUDAStatus = *unsafe { cuda.get(b"cuMemGetInfo_v2\0") }?;
    //let cuMemFreeAsync = *unsafe { cuda.get(b"cuMemFreeAsync\0") }?;
    let cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUDAStatus = *unsafe { cuda.get(b"cuMemFree_v2\0") }?;
    let cuMemcpyHtoDAsync: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUDAStatus =
        *unsafe { cuda.get(b"cuMemcpyHtoDAsync_v2\0") }?;
    //let cuMemcpyHtoD = *unsafe { cuda.get(b"cuMemcpyHtoD\0") }?;
    let cuMemcpyDtoHAsync: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUDAStatus =
        *unsafe { cuda.get(b"cuMemcpyDtoHAsync_v2\0") }?;
    //let cuMemcpyPeer = *unsafe { cuda.get(b"cuMemcpyPeer\0") }?;
    //let cuCtxSetCurrent = *unsafe { cuda.get(b"cuCtxGetCurrent\0") };
    //let cuCtxDestroy = *unsafe { cuda.get(b"cuCtxDestroy_v2\0") }?;
    let cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut CUmodule,
        *const c_void,
        c_uint,
        *mut CUjit_option,
        *mut *mut c_void,
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuModuleLoadDataEx\0") }?;
    let cuModuleGetFunction: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUDAStatus =
        *unsafe { cuda.get(b"cuModuleGetFunction\0") }?;
    let cuLaunchKernel: unsafe extern "C" fn(
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
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuLaunchKernel\0") }?;
    let cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUDAStatus = *unsafe { cuda.get(b"cuStreamCreate\0") }?;
    let cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus = *unsafe { cuda.get(b"cuStreamSynchronize\0") }?;
    let cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuStreamWaitEvent\0") }?;
    //let cuStreamDestroy = *unsafe { cuda.get(b"cuStreamDestroy\0") }?;
    let cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUDAStatus = *unsafe { cuda.get(b"cuModuleUnload\0") }?;
    let cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUDAStatus = *unsafe { cuda.get(b"cuEventCreate\0") }?;
    let cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUDAStatus = *unsafe { cuda.get(b"cuEventRecord\0") }?;
    let cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUDAStatus = *unsafe { cuda.get(b"cuEventSynchronize\0") }?;
    let cuEventQuery: unsafe extern "C" fn(CUevent) -> CUDAStatus = *unsafe { cuda.get(b"cuEventQuery\0") }?;
    let cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUDAStatus =
        *unsafe { cuda.get(b"cuEventElapsedTime\0") }?;
    let cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUDAStatus = *unsafe { cuda.get(b"cuEventDestroy\0") }?;
    let cuCtxEnablePeerAccess: unsafe extern "C" fn(CUcontext, c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuCtxEnablePeerAccess\0") }?;
    let cuMemcpyPeerAsync: unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize, CUstream) -> CUDAStatus =
        *unsafe { cuda.get(b"cuMemcpyPeerAsync\0") }?;
    let cuStreamBeginCapture: unsafe extern "C" fn(CUstream, CUstreamCaptureMode) -> CUDAStatus =
        *unsafe { cuda.get(b"cuStreamBeginCapture_v2\0") }?;
    let cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUDAStatus =
        *unsafe { cuda.get(b"cuStreamEndCapture\0") }?;
    let cuGraphInstantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, *mut c_void, *mut c_char, usize) -> CUDAStatus =
        *unsafe { cuda.get(b"cuGraphInstantiate_v2\0") }?;
    let cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUDAStatus = *unsafe { cuda.get(b"cuGraphLaunch\0") }?;
    let cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUDAStatus = *unsafe { cuda.get(b"cuGraphExecDestroy\0") }?;
    let cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUDAStatus = *unsafe { cuda.get(b"cuGraphDestroy\0") }?;
    let cuGraphExecUpdate: unsafe extern "C" fn(CUgraphExec, CUgraph, *mut CUgraphExecUpdateResultInfo) -> CUDAStatus =
        *unsafe { cuda.get(b"cuGraphExecUpdate_v2\0") }?;
    let cuMemcpyDtoDAsync: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUDAStatus =
        *unsafe { cuda.get(b"cuMemcpyDtoDAsync_v2\0") }?;
    //let cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUDAStatus = *unsafe { cuda.get(b"cuCtxDestroy_v2\0") }?;
    //let cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUDAStatus = *unsafe { cuda.get(b"cuDevicePrimaryCtxRetain\0") }?;

    if let Err(err) = unsafe { cuInit(0) }.check(ErrorStatus::Initialization) {
        if debug_dev {
            println!("[cuda] cuInit failed: {err:?}");
        }
        return Err(err);
    }
    let mut driver_version = 0;
    unsafe { cuDriverGetVersion(&raw mut driver_version) }.check(ErrorStatus::DeviceQuery)?;
    let mut num_devices = 0;
    unsafe { cuDeviceGetCount(&raw mut num_devices) }.check(ErrorStatus::DeviceQuery)?;
    if num_devices == 0 {
        return Err(BackendError { status: ErrorStatus::DeviceEnumeration, context: "[CUDA] no available device.".into() });
    }
    let device_ids: Vec<i32> =
        (0..num_devices).filter(|id| config.cuda.device_ids.as_ref().is_none_or(|ids| ids.contains(id))).collect();
    if debug_dev && !device_ids.is_empty() {
        println!(
            "[cuda] driver version {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }
    let fns = CudaFns {
        cuInit,
        cuDriverGetVersion,
        cuDeviceGetCount,
        cuDeviceGet,
        cuDeviceGetName,
        cuDeviceComputeCapability,
        cuDeviceTotalMem,
        cuDeviceGetAttribute,
        cuCtxCreate,
        cuMemAlloc,
        cuMemGetInfo,
        cuMemFree,
        cuMemcpyHtoDAsync,
        cuMemcpyDtoHAsync,
        cuModuleLoadDataEx,
        cuModuleGetFunction,
        cuLaunchKernel,
        cuStreamCreate,
        cuStreamSynchronize,
        cuStreamWaitEvent,
        cuModuleUnload,
        cuEventCreate,
        cuEventRecord,
        cuEventSynchronize,
        cuEventQuery,
        cuEventElapsedTime,
        cuEventDestroy,
        cuCtxEnablePeerAccess,
        cuMemcpyPeerAsync,
        cuStreamBeginCapture,
        cuStreamEndCapture,
        cuGraphInstantiate,
        cuGraphLaunch,
        cuGraphExecDestroy,
        cuGraphDestroy,
        cuGraphExecUpdate,
        cuMemcpyDtoDAsync,
    };
    let lib = Arc::new(cuda);
    Ok((lib, cudnn, device_ids, fns))
}

/// Spawns one GPU's worker thread (owns the CUDA context) and returns nothing;
/// all communication goes through the pool's channel. Moved verbatim out of
/// the old `initialize_device` so pool init can run without device init.
fn spawn_worker(
    fns: CudaFns,
    cudnn: Option<Arc<CudnnLib>>,
    lib: Arc<Library>,
    device: CUdevice,
    dev_ordinal: i32,
    debug_dev: bool,
    rx: Receiver<CUDACommand>,
    free_bytes_atomic: Arc<AtomicU64>,
    pool: Pool,
) {
    let CudaFns {
        cuCtxCreate,
        cuDeviceGetAttribute,
        cuEventCreate,
        cuEventDestroy,
        cuEventQuery,
        cuEventElapsedTime,
        cuEventRecord,
        cuEventSynchronize,
        cuCtxEnablePeerAccess,
        cuMemcpyPeerAsync,
        cuLaunchKernel,
        cuMemAlloc,
        cuMemFree,
        cuMemcpyDtoHAsync,
        cuMemcpyHtoDAsync,
        cuModuleGetFunction,
        cuModuleLoadDataEx,
        cuModuleUnload,
        cuStreamCreate,
        cuStreamSynchronize,
        cuStreamWaitEvent,
        cuStreamBeginCapture,
        cuStreamEndCapture,
        cuGraphInstantiate,
        cuGraphLaunch,
        cuGraphExecDestroy,
        cuGraphDestroy,
        cuGraphExecUpdate,
        cuMemcpyDtoDAsync,
        ..
    } = fns;
    std::thread::spawn(move || {
        let _lib = lib;
        //println!("INIT receiver");
        // Initialize raw CUDA context
        let mut context: CUcontext = ptr::null_mut();
        if let Err(e) = unsafe { cuCtxCreate(&raw mut context, 0, device) }.check(ErrorStatus::Initialization) {
            if debug_dev {
                println!("[cuda] context init failed: {e:?}");
            }
            return;
        }

        // Hardware queues (streams). Launches submit directly; graph
        // capture replays them via a captured CUDA graph.
        let queue_count = super::config().cuda.queues.unwrap_or(12);
        let mut streams: Vec<CUDAStream> = Vec::new();
        for _ in 0..queue_count {
            let mut stream = ptr::null_mut();
            if let Err(err) = unsafe { cuStreamCreate(&raw mut stream, 0) }.check(ErrorStatus::Initialization) {
                if debug_dev {
                    println!("[cuda] device {dev_ordinal}: stream init failed: {err:?}");
                }
                continue;
            }
            streams.push(CUDAStream { stream });
        }
        if streams.is_empty() {
            if debug_dev {
                println!("[cuda] device {dev_ordinal}: no streams created");
            }
            return;
        }

        // Per-axis max grid extents, checked against every evaluated
        // grid dimension before launching (see gws_from_kernel for the
        // compile-time counterpart covering constant dims).
        let mut max_grid = [0; 3];
        for (axis, attr) in [
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        ]
        .into_iter()
        .enumerate()
        {
            let mut value: c_int = 0;
            if let Err(err) = unsafe { (cuDeviceGetAttribute)(&raw mut value, attr, device) }.check(ErrorStatus::DeviceQuery) {
                if debug_dev {
                    println!("[cuda] device {dev_ordinal}: grid dim query failed: {err:?}");
                }
                return;
            }
            max_grid[axis] = i64::from(value);
        }

        let mut buffers: Slab<ChunkId, CUDABuffer> = Slab::new();
        // stack of buffers that are unused and can be reused
        let mut free_set: Set<ChunkId> = Set::default();
        let mut programs: Slab<DeviceProgramId, CUDAProgram> = Slab::new();
        // Captured partition executables by worker-side id. The
        // partition holds the live handle; a stale id recaptures, a
        // vars-mismatched id updates in place under the same id.
        let mut graphs: Slab<CudaGraphId, CapturedGraph> = Slab::new();
        // This worker's pool identity: placements resolve through the shard
        // addressed to this pool.
        let my_pool = pool;

        // Create the cuDNN handle on this worker thread (it owns the
        // current CUDA context). Optional — only used by AOT cudnn kernels.
        let cudnn_handle = cudnn.as_ref().and_then(|cudnn| {
            let mut handle = ptr::null_mut();
            if unsafe { (cudnn.create)(&raw mut handle) } == CUDNN_STATUS_SUCCESS {
                Some(handle)
            } else {
                None
            }
        });

        // Worker loop
        'work_thread_loop: while let Ok(cmd) = rx.recv() {
            match cmd {
                CUDACommand::Allocate { bytes, reply } => {
                    // Best-fit from the free list first (stable addresses
                    // across repeats — those bytes were already paid for);
                    // else fresh.
                    let best = free_set
                        .iter()
                        .filter_map(|id| {
                            let b = &buffers[*id];
                            (b.bytes >= bytes).then_some((b.bytes, *id))
                        })
                        .min();
                    if let Some((_, id)) = best {
                        free_set.remove(&id);
                        let _ = reply.send(Ok(id));
                        continue;
                    }
                    let mut ptr: CUdeviceptr = 0;
                    if let Err(err) = unsafe { (cuMemAlloc)(&raw mut ptr, bytes as usize) }.check(ErrorStatus::MemoryAllocation) {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                    assert!(ptr % 8 == 0, "Memory is not 8-byte aligned!");
                    debug_assert!(free_bytes_atomic.load(Ordering::SeqCst) > bytes as u64);
                    free_bytes_atomic.fetch_sub(bytes as u64, Ordering::SeqCst);
                    let buffer_id = buffers.push(CUDABuffer { ptr, bytes });
                    let _ = reply.send(Ok(buffer_id));
                }
                CUDACommand::Release { buffer_id } => {
                    // Put the id on the free list for stable-address reuse.
                    // Frees nothing: VRAM is reclaimed only by Dispose.
                    // The slab entry stays (tombstone) so ChunkIds are
                    // monotonic and never alias a different buffer.
                    if !buffers.contains_id(buffer_id) {
                        debug_assert!(false, "release of unknown buffer {buffer_id:?}");
                        continue;
                    }
                    debug_assert!(!free_set.contains(&buffer_id), "double release of buffer {buffer_id:?}");
                    free_set.insert(buffer_id);
                }
                CUDACommand::Dispose => {
                    todo!("Dispose drains the free list under a full device sync")
                }
                CUDACommand::TryReuse { buffer_ids, reply } => {
                    // All-or-nothing: every id must be on the free list, else
                    // some address moved and the graph must be retraced.
                    let ok = buffer_ids.iter().all(|id| free_set.contains(id));
                    if ok {
                        for id in &buffer_ids {
                            free_set.remove(id);
                        }
                    }
                    let _ = reply.send(Ok(ok));
                }
                CUDACommand::PoolToHost { src, dst, bytes, reply } => {
                    // Sync point: drain the (single) stream, then read back.
                    // No scheduler state: launches are submitted directly.
                    for st in &streams {
                        if let Err(err) = unsafe { (cuStreamSynchronize)(st.stream) }.check(ErrorStatus::MemoryCopyP2H) {
                            let _ = reply.send(Err(err));
                            continue 'work_thread_loop;
                        }
                    }
                    let Some(src_buf) = buffers.get(src) else {
                        let _ = reply.send(Err(BackendError {
                            status: ErrorStatus::MemoryCopyP2H,
                            context: "readback of unknown buffer".into(),
                        }));
                        continue;
                    };
                    // Never read past the device buffer's allocation (the
                    // requested length comes from the dst side, which may
                    // carry a larger padded allocation).
                    let bytes = bytes.min(src_buf.bytes);
                    if let Err(err) = unsafe { (cuMemcpyDtoHAsync)(dst.cast(), src_buf.ptr, bytes as usize, streams[0].stream) }
                        .check(ErrorStatus::MemoryCopyP2H)
                    {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                    if let Err(err) = unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::MemoryCopyP2H) {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                    let _ = reply.send(Ok(()));
                }
                CUDACommand::Compile { lws, gws, params, name, ptx, reply } => {
                    //println!("name {name}, gws {gws:?}, lws {lws:?} ptx:\n{}", std::ffi::CString::from_vec_with_nul(ptx.clone()).unwrap().into_string().unwrap());

                    let mut module = ptr::null_mut();
                    if let Err(err) =
                        unsafe { (cuModuleLoadDataEx)(&raw mut module, ptx.as_ptr().cast(), 0, ptr::null_mut(), ptr::null_mut()) }
                            .check(ErrorStatus::KernelCompilation)
                    {
                        if debug_dev {
                            println!("[cuda] PTX compilation failed: {err:?}");
                        }
                        //panic!();
                        _ = reply.send(Err(err));
                        continue;
                    }
                    let mut function: CUfunction = ptr::null_mut();
                    // Don't forget that the name is null terminated string
                    if let Err(err) = unsafe { (cuModuleGetFunction)(&raw mut function, module, name.as_ptr().cast()) }
                        .check(ErrorStatus::KernelLaunch)
                    {
                        if debug_dev {
                            println!("[cuda] kernel launch failed: {err:?}\n");
                        }
                        _ = reply.send(Err(err));
                        continue;
                    }

                    let program_id = programs.push(CUDAProgram::Module { module, function, lws, gws, params });
                    _ = reply.send(Ok(program_id));
                }
                CUDACommand::CompileCudnn { graph, reply } => {
                    let Some(cudnn) = &cudnn else {
                        _ = reply.send(Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "cuDNN library not loaded.".into(),
                        }));
                        continue;
                    };
                    match unsafe { build_cudnn_plan(cudnn, &graph, cuMemAlloc) } {
                        Ok(plan) => {
                            let program_id = programs.push(CUDAProgram::Cudnn { plan });
                            _ = reply.send(Ok(program_id));
                        }
                        Err(e) => {
                            _ = reply.send(Err(e));
                        }
                    }
                }
                CUDACommand::LaunchTimed { program_id, args, reply } => {
                    // Uncontended timing for autotune: drain the stream, then
                    // run the kernel solo bracketed by timing events.
                    for st in &streams {
                        if let Err(err) = unsafe { (cuStreamSynchronize)(st.stream) }.check(ErrorStatus::KernelSync) {
                            let _ = reply.send(Err(err));
                            continue 'work_thread_loop;
                        }
                    }
                    let mut start = ptr::null_mut();
                    let mut end = ptr::null_mut();
                    // Default event flags include timing support (needed by
                    // cuEventElapsedTime); blocking-sync-only (0x2) events
                    // used elsewhere do not record timestamps.
                    if let Err(err) = unsafe { (cuEventCreate)(&raw mut start, 0) }.check(ErrorStatus::KernelSync) {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                    if let Err(err) = unsafe { (cuEventCreate)(&raw mut end, 0) }.check(ErrorStatus::KernelSync) {
                        let _ = reply.send(Err(err));
                        continue;
                    }
                    let result = (|| -> Result<u64, BackendError> {
                        unsafe { (cuEventRecord)(start, streams[0].stream) }.check(ErrorStatus::KernelSync)?;
                        submit_launch(
                            &programs,
                            &buffers,
                            program_id,
                            &args,
                            streams[0].stream,
                            &max_grid,
                            &cudnn,
                            cudnn_handle,
                            cuLaunchKernel,
                        )?;
                        unsafe { (cuEventRecord)(end, streams[0].stream) }.check(ErrorStatus::KernelSync)?;
                        unsafe { (cuEventSynchronize)(end) }.check(ErrorStatus::KernelSync)?;
                        let mut ms = 0f32;
                        unsafe { (cuEventElapsedTime)(&raw mut ms, start, end) }.check(ErrorStatus::KernelSync)?;
                        Ok((ms * 1_000_000.0) as u64)
                    })();
                    let _ = unsafe { (cuEventDestroy)(start) };
                    let _ = unsafe { (cuEventDestroy)(end) };
                    let _ = reply.send(result);
                }
                CUDACommand::Replay { cmds, bound, vars, graph, reply } => {
                    let dry_run = std::env::var("ZYX_DRY_RUN").is_ok();
                    let result = (|| -> Result<ReplayResult, BackendError> {
                        let vars_map: Map<OpId, Constant> = vars.iter().copied().collect();
                        // Resident (chunk, offset) by slot, then fresh allocations.
                        let mut slot_chunk: Map<OpId, (ChunkId, usize)> =
                            bound.into_iter().map(|(s, c, o)| (s, (c, o))).collect();
                        let mut scalars: Map<OpId, Constant> = vars_map.clone();
                        let mut fresh: Vec<(OpId, ChunkId, usize)> = Vec::new();
                        for cmd in &cmds {
                            match cmd {
                                Cmd::Launch { outputs, .. } => {
                                    for (slot, dtype, dims) in outputs {
                                        if slot_chunk.contains_key(slot) {
                                            continue;
                                        }
                                        let bytes = dims
                                            .iter()
                                            .map(|d| d.eval(&vars_map))
                                            .fold(dtype.bit_size() as i64 / 8, |a, b| a * b);
                                        if bytes < 0 {
                                            return Err(BackendError {
                                                status: ErrorStatus::MemoryAllocation,
                                                context: format!("replay allocated negative bytes for {slot:?}").into(),
                                            });
                                        }
                                        // Stable-address reuse first: steady-state
                                        // replays land on the same chunks.
                                        let id = match free_set
                                            .iter()
                                            .filter_map(|bid| {
                                                let b = &buffers[*bid];
                                                (b.bytes >= bytes).then_some((b.bytes, *bid))
                                            })
                                            .min()
                                        {
                                            Some((_, id)) => {
                                                free_set.remove(&id);
                                                id
                                            }
                                            None => {
                                                let mut ptr: CUdeviceptr = 0;
                                                unsafe { (cuMemAlloc)(&raw mut ptr, bytes as usize) }
                                                    .check(ErrorStatus::MemoryAllocation)?;
                                                assert!(ptr % 8 == 0, "Memory is not 8-byte aligned!");
                                                free_bytes_atomic.fetch_sub(bytes as u64, Ordering::SeqCst);
                                                buffers.push(CUDABuffer { ptr, bytes })
                                            }
                                        };
                                        slot_chunk.insert(*slot, (id, 0));
                                        fresh.push((*slot, id, bytes as usize));
                                    }
                                }
                                Cmd::Alias { class, to } => {
                                    let region = *slot_chunk.get(to).ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelLaunch,
                                        context: format!("replay: alias target {to:?} is unplaced").into(),
                                    })?;
                                    slot_chunk.insert(*class, region);
                                }
                                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
                            }
                        }
                        // Slot resolution straight from the buffer slab.
                        // No Placement values are ever constructed here:
                        // they own their chunk and would release it.
                        let ptr_of = |slot: &OpId| -> Result<u64, BackendError> {
                            let (chunk, offset) = slot_chunk.get(slot).ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelLaunch,
                                context: format!("replay: launch slot {slot:?} is unplaced").into(),
                            })?;
                            buffers.get(*chunk).map(|b| b.ptr + *offset as u64).ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelLaunch,
                                context: format!("replay: unknown buffer for slot {slot:?}").into(),
                            })
                        };
                        // Captured params for one launch: pinned
                        // device-pointer slots plus boxed scalars (mirrors
                        // submit_launch's assembly; the caller holds the
                        // returned vec across the driver call).
                        let params_of = |slots: &[OpId],
                                         buffer_ptrs: &mut Vec<u64>,
                                         scalar_values: &mut Vec<Box<[u8]>>|
                         -> Result<Vec<*mut core::ffi::c_void>, BackendError> {
                            for slot in slots {
                                if slot_chunk.contains_key(slot) {
                                    buffer_ptrs.push(ptr_of(slot)?);
                                } else if !scalars.contains_key(slot) {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelLaunch,
                                        context: format!("replay: launch slot {slot:?} is neither placed nor bound").into(),
                                    });
                                }
                            }
                            let mut params: Vec<*mut core::ffi::c_void> = Vec::with_capacity(slots.len());
                            let mut buf_idx = 0usize;
                            for slot in slots {
                                if slot_chunk.contains_key(slot) {
                                    let ptr = &buffer_ptrs[buf_idx];
                                    buf_idx += 1;
                                    let at: *const u64 = core::ptr::from_ref(ptr);
                                    params.push(at.cast_mut().cast());
                                } else {
                                    scalar_values.push(scalars[slot].to_le_bytes().into());
                                    let value = scalar_values.last().unwrap();
                                    params.push(value.as_ptr().cast_mut().cast());
                                }
                            }
                            Ok(params)
                        };
                        // Grid + block for a Module program under current
                        // scalars, addressed by queue-local slot. Duplicates
                        // submit_launch's evaluation: that fn submits, here
                        // only the geometry is needed for capture-time
                        // recording.
                        let geometry_of =
                            |program_id: DeviceProgramId, slots: &[OpId]| -> Result<([u32; 3], [u32; 3]), BackendError> {
                                let CUDAProgram::Module { lws, gws, .. } = &programs[program_id] else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelLaunch,
                                        context: "geometry of non-module program".into(),
                                    });
                                };
                                let grid = |gdim: &GwsDim| -> Dim {
                                    gdim.eval(&mut |ordinal| {
                                        scalars
                                            .get(&slots[ordinal])
                                            .and_then(|c| c.as_dim())
                                            .expect("gws param must be a Variable slot")
                                    })
                                };
                                let default_gws = GwsDim::Const(1);
                                let (gx, gy, gz) = (
                                    grid(gws.first().unwrap_or(&default_gws)),
                                    grid(gws.get(1).unwrap_or(&default_gws)),
                                    grid(gws.get(2).unwrap_or(&default_gws)),
                                );
                                if gx < 0 || gy < 0 || gz < 0 || gx > max_grid[0] || gy > max_grid[1] || gz > max_grid[2] {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelLaunch,
                                        context: format!("grid dims ({gx},{gy},{gz}) exceed device max {max_grid:?}").into(),
                                    });
                                }
                                Ok(([gx as u32, gy as u32, gz as u32], [lws[0], lws[1], lws[2]]))
                            };
                        // cuDNN programs rebuild their variant pack per
                        // launch already — partitions holding one are
                        // submitted directly and never captured.
                        let has_cudnn = cmds.iter().any(|cmd| match cmd {
                            Cmd::Launch { program, .. } => matches!(programs[program.program_id], CUDAProgram::Cudnn { .. }),
                            _ => false,
                        });
                        if dry_run {
                            return Ok(ReplayResult { fresh, graph: None });
                        }
                        if has_cudnn {
                            // Direct submit, never captured: cuDNN rebuilds
                            // its variant pack per launch anyway.
                            let Some(cudnn) = cudnn.as_ref() else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: "cuDNN library not loaded.".into(),
                                });
                            };
                            let Some(handle) = cudnn_handle else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: "cuDNN handle missing.".into(),
                                });
                            };
                            for cmd in &cmds {
                                let Cmd::Launch { program, args, .. } = cmd else { continue };
                                match &programs[program.program_id] {
                                    CUDAProgram::Module { function, .. } => {
                                        let function = *function;
                                        let (grid, block) = geometry_of(program.program_id, args)?;
                                        let mut buffer_ptrs = Vec::new();
                                        let mut scalar_values = Vec::new();
                                        let mut kernel_params = params_of(args, &mut buffer_ptrs, &mut scalar_values)?;
                                        unsafe {
                                            (cuLaunchKernel)(
                                                function,
                                                grid[0],
                                                grid[1],
                                                grid[2],
                                                block[0],
                                                block[1],
                                                block[2],
                                                0,
                                                streams[0].stream,
                                                kernel_params.as_mut_ptr(),
                                                ptr::null_mut(),
                                            )
                                        }
                                        .check(ErrorStatus::KernelLaunch)?;
                                    }
                                    CUDAProgram::Cudnn { plan } => unsafe {
                                        let mut variant_pack = ptr::null_mut();
                                        if (cudnn.backend_create_descriptor)(
                                            CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                            &raw mut variant_pack,
                                        ) != CUDNN_STATUS_SUCCESS
                                        {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelLaunch,
                                                context: "cuDNN variant pack".into(),
                                            });
                                        }
                                        let mut data_ptrs: Vec<*mut c_void> = Vec::with_capacity(args.len());
                                        for slot in args {
                                            if slot_chunk.contains_key(slot) {
                                                data_ptrs.push(ptr_of(slot)? as *mut c_void);
                                            } else {
                                                let _ = (cudnn.backend_destroy_descriptor)(variant_pack);
                                                todo!("scalar variant-pack entries in cuDNN launches");
                                            }
                                        }
                                        let unique_ids: Vec<i64> = plan.arg_uids.clone();
                                        let set_ok = (cudnn.backend_set_attribute)(
                                            variant_pack,
                                            CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                            CUDNN_TYPE_INT64,
                                            unique_ids.len() as i64,
                                            unique_ids.as_ptr().cast(),
                                        ) == CUDNN_STATUS_SUCCESS
                                            && (cudnn.backend_set_attribute)(
                                                variant_pack,
                                                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                CUDNN_TYPE_VOID_PTR,
                                                data_ptrs.len() as i64,
                                                data_ptrs.as_ptr().cast(),
                                            ) == CUDNN_STATUS_SUCCESS
                                            && (if plan.workspace != 0 {
                                                let ws_ptr: *mut c_void = plan.workspace as *mut c_void;
                                                (cudnn.backend_set_attribute)(
                                                    variant_pack,
                                                    CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                    CUDNN_TYPE_VOID_PTR,
                                                    1,
                                                    (&raw const ws_ptr).cast(),
                                                ) == CUDNN_STATUS_SUCCESS
                                            } else {
                                                true
                                            })
                                            && (cudnn.backend_finalize)(variant_pack) == CUDNN_STATUS_SUCCESS;
                                        if set_ok {
                                            let _ = (cudnn.backend_execute)(handle, plan.plan, variant_pack);
                                        }
                                        let _ = (cudnn.backend_destroy_descriptor)(variant_pack);
                                        if !set_ok {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelLaunch,
                                                context: "cuDNN variant pack config".into(),
                                            });
                                        }
                                    },
                                }
                            }
                            return Ok(ReplayResult { fresh, graph: None });
                        }
                        // Fast tier: identical pointers and scalars — the
                        // executable is untouched, stream order alone
                        // serializes: relaunch with no other work.
                        if let Some(id) = graph
                            && let Some(g) = graphs.get(id)
                            && g.vars == vars
                        {
                            let mut current = Vec::with_capacity(g.ptrs.len());
                            for arg_slots in &g.node_args {
                                let mut buffer_ptrs: Vec<u64> = Vec::new();
                                let mut scalar_values: Vec<Box<[u8]>> = Vec::new();
                                params_of(arg_slots, &mut buffer_ptrs, &mut scalar_values)?;
                                current.extend_from_slice(&buffer_ptrs);
                            }
                            if current == g.ptrs {
                                unsafe { (cuGraphLaunch)(g.exec, streams[0].stream) }.check(ErrorStatus::KernelLaunch)?;
                                return Ok(ReplayResult { fresh, graph });
                            }
                        }
                        // TODO: multi-stream launch parallelism — assign
                        // independent launches to different streams by slot
                        // affinity (as OpenCL partitions do across queues).
                        // Capture, update, and relaunch all pin to
                        // streams[0] today, so launches serialize.
                        // Recapture: re-record the partition into a fresh
                        // temp graph under current addresses, grids, and
                        // scalars, plus the metadata for the fast tier.
                        let recaptured = (|| -> Result<(CUgraph, Vec<Vec<OpId>>, Vec<u64>), BackendError> {
                            unsafe { (cuStreamBeginCapture)(streams[0].stream, 0) }.check(ErrorStatus::KernelLaunch)?;
                            let mut node_args: Vec<Vec<OpId>> = Vec::new();
                            let mut ptrs: Vec<u64> = Vec::new();
                            for cmd in &cmds {
                                let Cmd::Launch { program, args, .. } = cmd else { continue };
                                // Capture path never holds cuDNN programs
                                // (those take the direct path above).
                                let CUDAProgram::Module { function, .. } = &programs[program.program_id] else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelLaunch,
                                        context: "capture of non-module program".into(),
                                    });
                                };
                                let function = *function;
                                let (grid, block) = geometry_of(program.program_id, args)?;
                                let mut buffer_ptrs = Vec::new();
                                let mut scalar_values = Vec::new();
                                let mut kernel_params = params_of(args, &mut buffer_ptrs, &mut scalar_values)?;
                                ptrs.extend_from_slice(&buffer_ptrs);
                                unsafe {
                                    (cuLaunchKernel)(
                                        function,
                                        grid[0],
                                        grid[1],
                                        grid[2],
                                        block[0],
                                        block[1],
                                        block[2],
                                        0,
                                        streams[0].stream,
                                        kernel_params.as_mut_ptr(),
                                        ptr::null_mut(),
                                    )
                                }
                                .check(ErrorStatus::KernelLaunch)?;
                                node_args.push(args.clone());
                            }
                            let mut graph_raw: CUgraph = ptr::null_mut();
                            unsafe { (cuStreamEndCapture)(streams[0].stream, &raw mut graph_raw) }
                                .check(ErrorStatus::KernelLaunch)?;
                            Ok((graph_raw, node_args, ptrs))
                        })();
                        // Update tier: an entry exists — swap the fresh
                        // recording into the live executable. The
                        // previous launch is async: drain first,
                        // updating a running executable is undefined.
                        if let Some(id) = graph
                            && graphs.contains_id(id)
                        {
                            unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::KernelSync)?;
                            let (tmp, node_args, ptrs) = match recaptured {
                                Ok(t) => t,
                                Err(e) => {
                                    graphs.remove(id);
                                    return Err(e);
                                }
                            };
                            let exec = graphs.get(id).expect("replay: cached graph is missing").exec;
                            let mut info = CUgraphExecUpdateResultInfo {
                                result: 0,
                                error_node: ptr::null_mut(),
                                error_from_node: ptr::null_mut(),
                            };
                            let updated =
                                unsafe { (cuGraphExecUpdate)(exec, tmp, &raw mut info) }.check(ErrorStatus::KernelLaunch).is_ok()
                                    && info.result == 0;
                            if updated {
                                unsafe { (cuGraphDestroy)(tmp) }.check(ErrorStatus::Deinitialization)?;
                                if let Some(g) = graphs.get_mut(id) {
                                    g.node_args = node_args;
                                    g.ptrs = ptrs;
                                    g.vars = vars.clone();
                                }
                                unsafe { (cuGraphLaunch)(exec, streams[0].stream) }.check(ErrorStatus::KernelLaunch)?;
                                return Ok(ReplayResult { fresh, graph });
                            }
                            // Topology changed: replace the executable
                            // under the same id, keep the temp only on
                            // success so a failure keeps no half state.
                            unsafe { (cuGraphExecDestroy)(exec) }.check(ErrorStatus::Deinitialization)?;
                            let mut new_exec: CUgraphExec = ptr::null_mut();
                            if let Err(e) =
                                unsafe { (cuGraphInstantiate)(&raw mut new_exec, tmp, ptr::null_mut(), ptr::null_mut(), 0) }
                                    .check(ErrorStatus::KernelLaunch)
                            {
                                unsafe { (cuGraphDestroy)(tmp) }.check(ErrorStatus::Deinitialization)?;
                                graphs.remove(id);
                                return Err(e);
                            }
                            unsafe { (cuGraphDestroy)(tmp) }.check(ErrorStatus::Deinitialization)?;
                            if let Some(g) = graphs.get_mut(id) {
                                *g = CapturedGraph { exec: new_exec, node_args, ptrs, vars: vars.clone() };
                            }
                            unsafe { (cuGraphLaunch)(new_exec, streams[0].stream) }.check(ErrorStatus::KernelLaunch)?;
                            return Ok(ReplayResult { fresh, graph });
                        }
                        // Capture path: no entry — instantiate the fresh
                        // recording and store it.
                        let (tmp, node_args, ptrs) = recaptured?;
                        let mut exec: CUgraphExec = ptr::null_mut();
                        if let Err(e) = unsafe { (cuGraphInstantiate)(&raw mut exec, tmp, ptr::null_mut(), ptr::null_mut(), 0) }
                            .check(ErrorStatus::KernelLaunch)
                        {
                            unsafe { (cuGraphDestroy)(tmp) }.check(ErrorStatus::Deinitialization)?;
                            return Err(e);
                        }
                        unsafe { (cuGraphDestroy)(tmp) }.check(ErrorStatus::Deinitialization)?;
                        let id = graphs.push(CapturedGraph { exec, node_args, ptrs, vars: vars.clone() });
                        unsafe { (cuGraphLaunch)(exec, streams[0].stream) }.check(ErrorStatus::KernelLaunch)?;
                        Ok(ReplayResult { fresh, graph: Some(id) })
                    })();
                    let _ = reply.send(result);
                }
                CUDACommand::CopyHtoD { src, dst, bytes, reply } => {
                    let result = (|| -> Result<(), BackendError> {
                        // Sync point: drain first so no in-flight writer
                        // aliases the fresh destination.
                        for st in &streams {
                            unsafe { (cuStreamSynchronize)(st.stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        let Some(dst_buf) = buffers.get(dst) else {
                            return Err(BackendError {
                                status: ErrorStatus::MemoryCopyP2P,
                                context: "upload into unknown buffer".into(),
                            });
                        };
                        let bytes = bytes.min(dst_buf.bytes);
                        if std::env::var("ZYX_DRY_RUN").is_err() {
                            unsafe { (cuMemcpyHtoDAsync)(dst_buf.ptr, src.cast(), bytes as usize, streams[0].stream) }
                                .check(ErrorStatus::MemoryCopyP2P)?;
                            unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        Ok(())
                    })();
                    let _ = reply.send(result);
                }
                CUDACommand::CopyDiskToD { ptr, extent, bytes, dst, reply } => {
                    let result = (|| -> Result<(), BackendError> {
                        for st in &streams {
                            unsafe { (cuStreamSynchronize)(st.stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        let Some(dst_buf) = buffers.get(dst) else {
                            return Err(BackendError {
                                status: ErrorStatus::MemoryCopyP2P,
                                context: "disk upload into unknown buffer".into(),
                            });
                        };
                        let bytes = bytes.min(dst_buf.bytes);
                        // No pinning: the driver rejects registering file
                        // mappings, and async DMA needs none (registration
                        // is bandwidth-only, a measured optimization later).
                        // The caller clamps to the mapped extent already.
                        debug_assert!(bytes <= extent, "disk upload past the mapped extent");
                        if std::env::var("ZYX_DRY_RUN").is_err() {
                            unsafe { (cuMemcpyHtoDAsync)(dst_buf.ptr, ptr.cast(), bytes as usize, streams[0].stream) }
                                .check(ErrorStatus::MemoryCopyP2P)?;
                            unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        Ok(())
                    })();
                    let _ = reply.send(result);
                }
                CUDACommand::CopyDtoD { src, dst, bytes, reply } => {
                    let result = (|| -> Result<(), BackendError> {
                        for st in &streams {
                            unsafe { (cuStreamSynchronize)(st.stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        let (src_ptr, dst_ptr) = match (buffers.get(src), buffers.get(dst)) {
                            (Some(s), Some(d)) => (s.ptr, d.ptr),
                            _ => {
                                return Err(BackendError {
                                    status: ErrorStatus::MemoryCopyP2P,
                                    context: "device copy of unknown buffer".into(),
                                });
                            }
                        };
                        if std::env::var("ZYX_DRY_RUN").is_err() {
                            unsafe { (cuMemcpyDtoDAsync)(dst_ptr, src_ptr, bytes as usize, streams[0].stream) }
                                .check(ErrorStatus::MemoryCopyP2P)?;
                            unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::MemoryCopyP2P)?;
                        }
                        Ok(())
                    })();
                    let _ = reply.send(result);
                }
                CUDACommand::DestroyGraph { id } => {
                    // Tolerant: an unknown id is already gone (recapture
                    // replaced it, or the worker never stored it). Drain:
                    // the handle may drop while its launch is in flight.
                    let _ = unsafe { (cuStreamSynchronize)(streams[0].stream) }.check(ErrorStatus::KernelSync);
                    if graphs.contains_id(id) {
                        let old = unsafe { graphs.remove_and_return(id) };
                        let _ = unsafe { (cuGraphExecDestroy)(old.exec) }.check(ErrorStatus::Deinitialization);
                    }
                }
                CUDACommand::ReleaseProgram { program_id } => {
                    match &programs[program_id] {
                        CUDAProgram::Module { module, .. } => {
                            let _ = unsafe { (cuModuleUnload)(*module) }.check(ErrorStatus::Deinitialization);
                        }
                        CUDAProgram::Cudnn { plan } => {
                            if let Some(cudnn) = &cudnn {
                                for desc in plan.descrs.iter().rev() {
                                    let _ = unsafe { (cudnn.backend_destroy_descriptor)(*desc) };
                                }
                                if plan.workspace != 0 {
                                    let _ = unsafe { (cuMemFree)(plan.workspace) }.check(ErrorStatus::MemoryDeallocation);
                                }
                            }
                        }
                    }
                    programs.remove(program_id);
                }
            }
        }
        //println!("DEINIT receiver");
    });
}

/// Single backend initializer: enumerates GPUs once, spawns one worker per
/// GPU, and builds both the pool and device tables in the same loop.
/// Reads config directly; called via `backend()` which publishes both tables.
fn initialize_backend() -> Result<(Vec<Mutex<CUDAMemoryPool>>, Vec<Mutex<CUDADevice>>), BackendError> {
    let debug_dev = super::debug_backends();
    let (lib, cudnn, device_ids, fns) = load_driver()?;
    let cudnn_available = cudnn.is_some();
    let mut pools = Vec::new();
    let mut devs = Vec::new();
    for dev_ordinal in device_ids {
        let mut device = 0;
        if let Err(err) = unsafe { (fns.cuDeviceGet)(&raw mut device, dev_ordinal) }.check(ErrorStatus::DeviceEnumeration) {
            if debug_dev {
                println!("[cuda] device {dev_ordinal}: could not be enumerated: {err:?}.");
            }
            continue;
        }
        let mut device_name = [0; 100];
        let Ok(()) = unsafe { (fns.cuDeviceGetName)(device_name.as_mut_ptr(), 100, device) }.check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        let Ok(()) =
            unsafe { (fns.cuDeviceComputeCapability)(&raw mut major, &raw mut minor, device) }.check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        if debug_dev {
            println!("[cuda] {:?}, compute: {major}.{minor}", unsafe { std::ffi::CStr::from_ptr(device_name.as_ptr()) });
        }
        let mut free_bytes = 0usize;
        let Ok(()) = unsafe { (fns.cuDeviceTotalMem)(&raw mut free_bytes, device) }.check(ErrorStatus::DeviceQuery) else {
            continue;
        };
        if debug_dev {
            println!("[cuda] device total memory: {} MB", free_bytes / (1024 * 1024));
        }
        let (tx, rx): (Sender<CUDACommand>, Receiver<CUDACommand>) = channel();
        let free_bytes_atomic = Arc::new(AtomicU64::new(free_bytes as u64));
        let index = devs.len();
        spawn_worker(
            fns,
            cudnn.clone(),
            Arc::clone(&lib),
            device,
            dev_ordinal,
            debug_dev,
            rx,
            Arc::clone(&free_bytes_atomic),
            Pool::Cuda(index as u16),
        );
        pools.push(Mutex::new(CUDAMemoryPool {
            tx: tx.clone(),
            free_bytes: free_bytes_atomic,
            device,
            dev_ordinal,
            fns,
            cudnn_available,
            _lib: Arc::clone(&lib),
        }));
        let mut dev = CUDADevice {
            tx,
            device,
            dev_info: Arc::new(DeviceInfo {
                compute: 1024 * 1024 * 1024 * 1024,
                max_global_work_dims: vec![64, 64, 64],
                max_local_threads: 1,
                max_local_work_dims: vec![1, 1, 1],
                local_mem_size: 0,
                max_register_bytes: 1024,
                preferred_vector_size: 16,
                tensor_cores: major >= 7,
                warp_size: 32,
                cc: [major, minor],
                dtype_capability: [DTypeCapability::none(); DType::N_DTYPES],
                has_native_exp2: true,
                supported_vec_lens: vec![],
                tenstorrent: false,
                tile: [1, 1],
                tile_sizes: vec![],
                wmma_layouts: if major >= 7 { vec![MMADims::m16n8k8] } else { vec![] },
                num_circular_buffers: 0,
                has_openmp: false,
            }),
            memory_pool: Pool::Cuda(index as u16),
            compute_capability: [major, minor],
            cudnn_available,
            dev_id: u32::try_from(dev_ordinal).unwrap(),
        };
        let cuDeviceGetAttribute = fns.cuDeviceGetAttribute;
        let max_regs_per_block: i32 =
            dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cuDeviceGetAttribute)?;
        let max_threads_per_block: i32 =
            dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDeviceGetAttribute)?;
        dev.dev_info = Arc::new(DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024, // TODO run a kernel to get an estimate
            max_global_work_dims: vec![
                Dim::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cuDeviceGetAttribute)?).unwrap(),
                Dim::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, cuDeviceGetAttribute)?).unwrap(),
                Dim::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, cuDeviceGetAttribute)?).unwrap(),
            ],
            max_local_threads: u32::try_from(max_threads_per_block).unwrap(),
            max_local_work_dims: vec![
                u32::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, cuDeviceGetAttribute)?).unwrap(),
                u32::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, cuDeviceGetAttribute)?).unwrap(),
                u32::try_from(dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, cuDeviceGetAttribute)?).unwrap(),
            ],
            local_mem_size: Dim::try_from(
                dev.get(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDeviceGetAttribute)?,
            )
            .unwrap(),
            max_register_bytes: (max_regs_per_block as i64 / max_threads_per_block as i64).min(256) * 4,
            preferred_vector_size: 16,
            tensor_cores: major >= 7,
            warp_size: 32,
            cc: [major, minor],
            dtype_capability: {
                let mut capability = [DTypeCapability::all(); DType::N_DTYPES];
                if major < 8 {
                    capability[DType::BF16 as usize] = DTypeCapability::none();
                }
                capability[DType::F16 as usize] =
                    capability[DType::F16 as usize].exclude(DTypeCapability::LOG2 | DTypeCapability::SIN | DTypeCapability::SQRT);
                capability[DType::F64 as usize] =
                    capability[DType::F64 as usize].exclude(DTypeCapability::LOG2 | DTypeCapability::SIN | DTypeCapability::SQRT);
                capability
            },
            has_native_exp2: true,
            supported_vec_lens: vec![],
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: if major >= 7 { vec![MMADims::m16n8k8] } else { vec![] },
            num_circular_buffers: 0,
            has_openmp: false,
        });
        devs.push(Mutex::new(dev));
    }
    Ok((pools, devs))
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<CUDADevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::Cuda({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}

impl CUDAMemoryPool {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub fn free_bytes(&self) -> Dim {
        self.free_bytes.load(Ordering::SeqCst) as i64
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        if bytes > self.free_bytes.load(Ordering::SeqCst) as i64 {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "Allocation failure.".into() });
        }
        let (reply, reply_rx) = channel();
        self.tx.send(CUDACommand::Allocate { bytes, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Tries to reuse existing allocations, all or nothing, returns true if possible, otherwise returns false
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let (reply, reply_rx) = channel();
        self.tx.send(CUDACommand::TryReuse { buffer_ids: buffer_ids.clone(), reply }).unwrap();
        reply_rx.recv().unwrap().unwrap_or(false)
    }

    /// Put single buffer into the free list
    pub fn release(&mut self, buffer_id: ChunkId) {
        self.tx.send(CUDACommand::Release { buffer_id }).unwrap();
    }

    /// Free all buffers in the free list at once
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn dispose(&mut self) {
        self.tx.send(CUDACommand::Dispose).unwrap();
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(CUDACommand::PoolToHost { src, dst: dst.as_mut_ptr(), bytes: dst.len() as i64, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Direct device->host DMA into caller-owned memory — NO staging buffer.
    /// The caller must keep the destination allocation alive (and untouched)
    /// until the reply arrives.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host_ptr(&mut self, src: ChunkId, dst: *mut u8, bytes: i64) -> Result<(), BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(CUDACommand::PoolToHost { src, dst: dst.cast(), bytes, reply }).unwrap();
        reply_rx.recv().unwrap()
    }
}

impl CUDADevice {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub fn info(&self) -> Arc<DeviceInfo> {
        self.dev_info.clone()
    }

    pub const fn memory_pool(&self) -> Pool {
        self.memory_pool
    }

    pub fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let rendered = kernel.render()?;
        let mut order = rendered.ops_in_order();
        let Some(GPUOp::Params(params)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "head op is not Params".into() });
        };
        let Some(GPUOp::Grid(gws)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "second op is not Grid".into() });
        };
        let Some(GPUOp::LocalWorkSize(lws)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "third op is not LocalWorkSize".into() });
        };
        let Some(fourth) = order.next() else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "descriptor ends before fourth op".into(),
            });
        };
        let lws: [u32; 3] = *lws;
        let mut name = format!("k_{}", lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),);
        // PTX-direct descriptor (ZYX_PTX render): assembled bytes ride
        // the descriptor and skip NVRTC entirely.
        if let Op::PTX(ptx) = fourth {
            // Single-variant enum: irrefutable today, and adding a
            // variant breaks this `let` at compile time, forcing
            // decode handling.
            let PTXOp::Bytes(bytes) = ptx.as_ref();
            let ptx_vec: Vec<u8> = bytes.to_vec();
            name += "\0";
            let (reply, reply_rx) = channel();
            self.tx
                .send(CUDACommand::Compile {
                    lws,
                    gws: gws.to_vec(),
                    params: params.to_vec(),
                    name: name.into_boxed_str(),
                    ptx: ptx_vec,
                    reply,
                })
                .unwrap();
            return reply_rx.recv().unwrap();
        }
        let Op::Source(source) = fourth else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "fourth op is neither Source nor PTX".into(),
            });
        };
        let source = source.as_str().to_string();
        if debug_asm {
            println!();
            println!("{source}");
        }

        let cudartc_paths = [
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so",
            "/opt/cuda/lib64/libnvrtc.so",
            "/opt/cuda/targets/x86_64-linux/lib/libnvrtc.so",
            "/lib/x86_64-linux-gnu/libnvrtc.so",
            "/usr/lib/libnvrtc.so",
            "/usr/lib64/libnvrtc.so",
        ];
        let cudartc = cudartc_paths.iter().find_map(|&path| unsafe { Library::new(path) }.ok());
        let Some(cudartc) = cudartc else {
            return Err(BackendError { status: ErrorStatus::Initialization, context: "[CUDA] libnvrtc.so not found.".into() });
        };
        let nvrtcCreateProgram: unsafe extern "C" fn(
            *mut nvrtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> nvrtcResult = *unsafe { cudartc.get(b"nvrtcCreateProgram\0") }.unwrap();
        let nvrtcCompileProgram: unsafe extern "C" fn(nvrtcProgram, c_int, *const *const c_char) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcCompileProgram\0") }.unwrap();
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
                &raw mut program,
                source.as_ptr().cast(),
                name.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelCompilation)?;

        let mut opts = vec![
            "--use_fast_math".into(),
            format!("--gpu-architecture=compute_{}{}", self.compute_capability[0], self.compute_capability[1]),
        ];

        let include_paths = [
            "/usr/include",
            "/usr/local/cuda/include",
            "/opt/cuda/targets/x86_64-linux/include",
        ];
        let mut include_path: Option<PathBuf> = None;
        for path in include_paths {
            let mut path_buf = PathBuf::from(path);
            path_buf.push("cuda_fp16.h");
            if path_buf.exists() {
                include_path = Some(PathBuf::from(path));
                break;
            }
        }
        if include_path.is_none() {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "[cuda] cuda_fp16.h not found".into() });
        }

        if let Some(path) = include_path {
            let path = format!("--include-path={}", path.display());
            opts.push(path);
        }
        // Because rust
        let opts_cstrings: Vec<CString> = opts.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();

        let opts: Vec<*const i8> = opts_cstrings.iter().map(|c| c.as_ptr()).collect();

        if let Err(e) =
            unsafe { nvrtcCompileProgram(program, opts.len() as i32, opts.as_ptr()) }.check(ErrorStatus::KernelCompilation)
        {
            println!("[CUDA] compilation error {e:?}");
            let mut program_log_size: usize = 0;
            unsafe { nvrtcGetProgramLogSize(program, &raw mut program_log_size) }.check(ErrorStatus::KernelCompilation)?;
            let mut program_log_vec: Vec<u8> = vec![0; program_log_size + 1];
            unsafe { nvrtcGetProgramLog(program, program_log_vec.as_mut_ptr().cast()) }.check(ErrorStatus::KernelCompilation)?;
            println!("[CUDA] {}", String::from_utf8_lossy(&program_log_vec));
        }
        let mut ptx_size: usize = 0;
        unsafe { nvrtcGetPTXSize(program, &raw mut ptx_size) }.check(ErrorStatus::KernelCompilation)?;
        let mut ptx_vec: Vec<u8> = vec![0; ptx_size];
        unsafe { nvrtcGetPTX(program, ptx_vec.as_mut_ptr().cast()) }.check(ErrorStatus::KernelCompilation)?;
        unsafe { nvrtcDestroyProgram(&raw mut program) }.check(ErrorStatus::KernelCompilation)?;

        name += "\0";
        let (reply, reply_rx) = channel();
        self.tx
            .send(CUDACommand::Compile {
                lws,
                gws: gws.to_vec(),
                params: params.to_vec(),
                name: name.into_boxed_str(),
                ptx: ptx_vec,
                reply,
            })
            .unwrap();
        reply_rx.recv().unwrap()
    }

    /// Timed launch for autotune. Returns the kernel's run time in nanos,
    /// measured uncontended: the worker drains all queues, then runs this
    /// kernel solo bracketed by timing events.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(CUDACommand::LaunchTimed { program_id, args: args.into(), reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.tx.send(CUDACommand::ReleaseProgram { program_id }).unwrap();
    }

    /// Pattern-matches matmul subgraphs and JIT-compiles a cuDNN execution plan
    /// for each with the exact shapes and dtypes, adding `Node::Kernel`s (with
    /// `time = 1`) so they beat any fused zyx kernel in extraction. Only f32 is
    /// supported for now (compute type float).
    pub fn match_graph(&mut self, graph: &mut Graph, outputs: &BTreeSet<OpId>) {
        if !self.cudnn_available {
            return;
        }
        let order = graph.topo_sort_classes::<true>(&Set::default(), outputs, None);
        for &cid in &order {
            let Some(mm) = graph.match_matmul(cid) else {
                continue;
            };
            // Only f32 matmul is supported by the cudnn matmul builder for now.
            if mm.in_dtype != DType::F32 || mm.acc_dtype != DType::F32 {
                continue;
            }
            let [m, n, k] = [mm.m, mm.n, mm.k];
            println!("[CUDA] cuDNN matched matmul m={m}, n={n}, k={k}");

            let graph_desc = CudnnGraph {
                tensors: vec![
                    CudnnTensor { uid: 0, shape: vec![m, k], dtype: DType::F32, is_virtual: false },
                    CudnnTensor { uid: 1, shape: vec![k, n], dtype: DType::F32, is_virtual: false },
                    CudnnTensor { uid: 2, shape: vec![m, n], dtype: DType::F32, is_virtual: false },
                ],
                ops: vec![CudnnOp::Matmul { a: 0, b: 1, c: 2, compute_dtype: DType::F32 }],
                arg_uids: vec![0, 1, 2],
            };
            let (reply, reply_rx) = channel();
            self.tx.send(CUDACommand::CompileCudnn { graph: graph_desc, reply }).unwrap();
            let Ok(program_id) = reply_rx.recv().unwrap() else {
                continue;
            };
            let inputs = graph.push_op(Op::Stack { ops: Box::new([mm.a, mm.b]) });
            let outputs = graph.push_op(Op::Stack { ops: Box::new([mm.out]) });
            graph.mint_node(
                Op::Program {
                    inputs,
                    outputs,
                    info: Box::new((
                        ProgramId {
                            dev: match self.memory_pool {
                                Pool::Cuda(i) => Dev::Cuda(i),
                                pool => unreachable!("CUDA device with non-CUDA pool {pool:?}"),
                            },
                            program_id,
                        },
                        1,
                    )),
                },
                mm.out,
            );
        }
    }
}

/// Preplanned CUDA partition: the ordered commands plus per-command death
/// lists plus the worker-side captured graph, if this partition has run
/// before. Replay ships the whole partition to the worker in one roundtrip
/// (bound slots, scalars, cached-graph hint); the worker allocates unbound
/// defs up front — nothing allocates inside the captured region — captures
/// on first sight, updates + relaunches on repeats. The graph slot is
/// interior-mutable: plans replay through shared references, and replacing
/// the handle destroys the old executable on the worker via `Drop`.
#[derive(Debug)]
pub(crate) struct CudaPartition {
    cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    pub(crate) dev: u16,
    graph: Mutex<Option<CudaGraph>>,
}

impl CUDADevice {
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>, dev: u16) -> CudaPartition {
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
        CudaPartition { cmds, deaths, dev, graph: Mutex::new(None) }
    }

    /// Copy executing a transfer into this device's pool: host uploads go
    /// through a blocking HtoD, same-pool pairs through a blocking DtoD,
    /// everything else is later work. Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        debug_assert!(bytes >= 0, "CUDA copy of negative bytes");
        let [src_shard] = &src.shards[..] else {
            todo!("CUDA copy of multi-shard source placement")
        };
        let [dst_shard] = &dst.shards[..] else {
            todo!("CUDA copy of multi-shard destination placement")
        };
        debug_assert_eq!(dst_shard.pool, self.memory_pool, "CUDA copy destination is not on this device");
        let dead = |_| BackendError { status: ErrorStatus::MemoryCopyP2P, context: "cuda worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::MemoryCopyP2P,
            context: "cuda worker hung up".into(),
        };
        match src_shard.pool {
            p if p == self.memory_pool => {
                let (reply, reply_rx) = channel();
                self.tx.send(CUDACommand::CopyDtoD { src: src_shard.chunk, dst: dst_shard.chunk, bytes, reply }).map_err(dead)?;
                return reply_rx.recv().map_err(dead_rx)?;
            }
            Pool::Host => {
                // The host lock is held across the blocking roundtrip: the
                // source pointer stays valid. The worker never takes it.
                let host = super::host::pool();
                let pool = super::lock(Pool::Host, host);
                let src_ptr = pool.get_buffer(src_shard.chunk).as_ptr();
                let (reply, reply_rx) = channel();
                self.tx.send(CUDACommand::CopyHtoD { src: src_ptr, dst: dst_shard.chunk, bytes, reply }).map_err(dead)?;
                return reply_rx.recv().map_err(dead_rx)?;
            }
            #[cfg(unix)]
            Pool::Disk => {
                // Direct DMA from the file mapping, executed fully on the
                // worker (pin bracketing needs the worker's CUDA context).
                // The extent holds exact tensor bytes; the destination is
                // over-allocated (one extra element) — never read past the
                // extent.
                let disk = super::disk::pool();
                let dpool = super::lock(Pool::Disk, disk);
                let (ptr, extent) = dpool.mapped_ptr(src_shard.chunk);
                let n = bytes.min(extent);
                let (reply, reply_rx) = channel();
                self.tx.send(CUDACommand::CopyDiskToD { ptr, extent, bytes: n, dst: dst_shard.chunk, reply }).map_err(dead)?;
                let res = reply_rx.recv().map_err(dead_rx)?;
                return res;
            }
            #[cfg(windows)]
            Pool::Disk => todo!("CUDA copy from disk on windows"),
            p => todo!("CUDA copy from {p:?}"),
        }
    }
}

impl CudaPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut CUDADevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        let my_pool = dev.memory_pool;
        let mut bound = Vec::with_capacity(resolved.len());
        for (slot, placement) in resolved.iter() {
            let Some(shard) = placement.shards.iter().find(|s| s.pool == my_pool) else {
                continue;
            };
            bound.push((*slot, shard.chunk, shard.offset));
        }
        // Sorted: the worker compares vars vectors directly for the
        // fast-vs-update decision.
        let mut vars_vec: Vec<(OpId, Constant)> = vars.iter().map(|(s, c)| (*s, *c)).collect();
        vars_vec.sort_unstable_by_key(|(s, _)| *s);
        let cached = self.graph.lock().expect("cuda partition graph slot poisoned").as_ref().map(|g| g.id);
        let (reply, reply_rx) = channel();
        let dead = |_| BackendError { status: ErrorStatus::KernelLaunch, context: "cuda worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: "cuda worker hung up".into(),
        };
        dev.tx
            .send(CUDACommand::Replay { cmds: self.cmds.clone(), bound, vars: vars_vec, graph: cached, reply })
            .map_err(dead)?;
        let res = reply_rx.recv().map_err(dead_rx)??;
        if let Some(id) = res.graph {
            let mut slot = self.graph.lock().expect("cuda partition graph slot poisoned");
            // Repeats return the id they were given: only a fresh id
            // replaces the handle (replacing with the same id would
            // destroy the live executable via `Drop`).
            if slot.as_ref().is_none_or(|g| g.id != id) {
                *slot = Some(CudaGraph { id, tx: dev.tx.clone() });
            }
        }
        for (slot, chunk, len) in res.fresh {
            resolved.insert(slot, Arc::new(Placement { shards: vec![Shard { pool: my_pool, chunk, offset: 0, len }] }));
        }
        for dead in self.deaths.iter().flatten() {
            resolved.remove(dead);
        }
        Ok(())
    }
}

/// Resolves a launch arg region to a device pointer: chunk base plus the
/// region offset. Debug-checked against the allocation size — the region
/// must lie inside the chunk (best-fit slack is allowed, overrun is not).
fn buffer_ptr(buffers: &Slab<ChunkId, CUDABuffer>, chunk: ChunkId, offset: usize, len: usize) -> Result<u64, BackendError> {
    let buf = buffers
        .get(chunk)
        .ok_or(BackendError { status: ErrorStatus::KernelLaunch, context: "launch arg addresses an unknown buffer".into() })?;
    debug_assert!((offset + len) as u64 <= buf.bytes as u64, "launch arg region overruns its chunk");
    Ok(buf.ptr + offset as u64)
}

/// Resolves launch args against the buffer table and submits one kernel (or
/// cuDNN plan) onto `stream`. Used by `LaunchTimed`.
#[allow(clippy::too_many_arguments)]
fn submit_launch(
    programs: &Slab<DeviceProgramId, CUDAProgram>,
    buffers: &Slab<ChunkId, CUDABuffer>,
    program_id: DeviceProgramId,
    args: &[LaunchArg],
    stream: CUstream,
    max_grid: &[i64; 3],
    cudnn: &Option<Arc<CudnnLib>>,
    cudnn_handle: Option<cudnnHandle_t>,
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
) -> Result<(), BackendError> {
    match &programs[program_id] {
        CUDAProgram::Module { function, lws, gws, .. } => {
            let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
            // Boxed so reallocs of this vec can never dangle the
            // pointers handed to cuLaunchKernel.
            let mut scalar_values: Vec<Box<[u8]>> = Vec::new();
            // Stable storage for the device pointers of buffer args —
            // cuLaunchKernel receives their addresses by reference.
            let mut buffer_ptrs: Vec<u64> = Vec::with_capacity(args.len());
            for arg in args.iter() {
                match arg {
                    LaunchArg::Buffer { chunk, offset, len } => buffer_ptrs.push(buffer_ptr(buffers, *chunk, *offset, *len)?),
                    LaunchArg::Variable(_) => {}
                }
            }
            let mut buf_ptr_idx = 0usize;
            for arg in args.iter() {
                match arg {
                    LaunchArg::Buffer { .. } => {
                        let ptr = &buffer_ptrs[buf_ptr_idx];
                        buf_ptr_idx += 1;
                        let slot: *const u64 = core::ptr::from_ref(ptr);
                        kernel_params.push(slot.cast_mut().cast());
                    }
                    LaunchArg::Variable(constant) => {
                        scalar_values.push(constant.to_le_bytes().into());
                        let value = scalar_values.last().unwrap();
                        kernel_params.push(value.as_ptr().cast_mut().cast());
                    }
                }
            }
            let grid = |gdim: &GwsDim| -> Dim {
                gdim.eval(&mut |ordinal| match &args[ordinal] {
                    LaunchArg::Variable(c) => c.as_dim().unwrap(),
                    LaunchArg::Buffer { .. } => unreachable!("gws param must be a Variable launch arg"),
                })
            };
            let default_gws = GwsDim::Const(1);
            let (gx, gy, gz) = (
                grid(gws.first().unwrap_or(&default_gws)),
                grid(gws.get(1).unwrap_or(&default_gws)),
                grid(gws.get(2).unwrap_or(&default_gws)),
            );
            if gx < 0 || gy < 0 || gz < 0 || gx > max_grid[0] || gy > max_grid[1] || gz > max_grid[2] {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("grid dims ({gx},{gy},{gz}) exceed device max {max_grid:?}").into(),
                });
            }
            let (gx, gy, gz) = (u32::try_from(gx).unwrap(), u32::try_from(gy).unwrap(), u32::try_from(gz).unwrap());
            if crate::debug_mask().launch() {
                println!("[cuda] launch {program_id:?} grid=({gx},{gy},{gz}) args={args:?}");
            }
            unsafe {
                (cuLaunchKernel)(
                    *function,
                    gx,
                    gy,
                    gz,
                    lws[0],
                    lws[1],
                    lws[2],
                    0,
                    stream,
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            }
            .check(ErrorStatus::KernelLaunch)
        }
        CUDAProgram::Cudnn { plan } => unsafe { launch_cudnn_plan(cudnn, cudnn_handle, plan, buffers, args, stream) },
    }
}

/// dlopens libcudnn.so.9 and resolves the graph API symbols. Returns `None` if
/// the library or any required symbol is missing — cuDNN AOT is then disabled.
fn load_cudnn() -> Option<Arc<CudnnLib>> {
    let paths = [
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
        "/usr/local/lib/libcudnn.so.9",
        "/usr/lib/libcudnn.so.9",
        "/usr/local/cuda/lib64/libcudnn.so.9",
        "/opt/cuda/lib64/libcudnn.so.9",
        "/opt/cudnn/lib/libcudnn.so.9",
    ];
    let lib = paths.into_iter().find_map(|path| unsafe { Library::new(path) }.ok())?;
    let create: unsafe extern "C" fn(*mut cudnnHandle_t) -> cudnnStatus_t = *unsafe { lib.get(b"cudnnCreate\0") }.ok()?;
    let destroy: unsafe extern "C" fn(cudnnHandle_t) -> cudnnStatus_t = *unsafe { lib.get(b"cudnnDestroy\0") }.ok()?;
    let backend_create_descriptor: unsafe extern "C" fn(c_int, *mut cudnnBackendDescriptor_t) -> cudnnStatus_t =
        *unsafe { lib.get(b"cudnnBackendCreateDescriptor\0") }.ok()?;
    let backend_destroy_descriptor: unsafe extern "C" fn(cudnnBackendDescriptor_t) -> cudnnStatus_t =
        *unsafe { lib.get(b"cudnnBackendDestroyDescriptor\0") }.ok()?;
    let backend_finalize: unsafe extern "C" fn(cudnnBackendDescriptor_t) -> cudnnStatus_t =
        *unsafe { lib.get(b"cudnnBackendFinalize\0") }.ok()?;
    let backend_set_attribute: unsafe extern "C" fn(cudnnBackendDescriptor_t, c_int, c_int, i64, *const c_void) -> cudnnStatus_t =
        *unsafe { lib.get(b"cudnnBackendSetAttribute\0") }.ok()?;
    let backend_get_attribute: unsafe extern "C" fn(
        cudnnBackendDescriptor_t,
        c_int,
        c_int,
        i64,
        *mut i64,
        *mut c_void,
    ) -> cudnnStatus_t = *unsafe { lib.get(b"cudnnBackendGetAttribute\0") }.ok()?;
    let backend_execute: unsafe extern "C" fn(
        cudnnHandle_t,
        cudnnBackendDescriptor_t,
        cudnnBackendDescriptor_t,
    ) -> cudnnStatus_t = *unsafe { lib.get(b"cudnnBackendExecute\0") }.ok()?;
    Some(Arc::new(CudnnLib {
        _lib: lib,
        create,
        destroy,
        backend_create_descriptor,
        backend_destroy_descriptor,
        backend_finalize,
        backend_set_attribute,
        backend_get_attribute,
        backend_execute,
    }))
}

/// Maps a zyx `DType` to a cuDNN `cudnnDataType_t` value.
fn cudnn_data_type(dtype: DType) -> Option<c_int> {
    Some(match dtype {
        DType::F32 => CUDNN_DATA_FLOAT,
        DType::F16 => CUDNN_DATA_HALF,
        DType::BF16 => CUDNN_DATA_BFLOAT16,
        _ => return None,
    })
}

/// JIT-compiles a cuDNN execution plan for a generic [`CudnnGraph`] subgraph:
/// tensor descriptors are built for every tensor, each op is expanded into its
/// cuDNN op descriptor (via [`build_cudnn_op`]), engine heuristics pick a
/// kernel, and the execution plan is finalized with fixed shapes. Returns the
/// compiled plan with its workspace allocated.
#[allow(clippy::too_many_lines)]
unsafe fn build_cudnn_plan(
    cudnn: &CudnnLib,
    graph: &CudnnGraph,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus,
) -> Result<CudnnPlan, BackendError> {
    unsafe {
        let mut descrs: Vec<cudnnBackendDescriptor_t> = Vec::new();
        let mut tensor_descs: Vec<cudnnBackendDescriptor_t> = Vec::new();
        for tensor in &graph.tensors {
            let mut desc = ptr::null_mut();
            if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &raw mut desc) != CUDNN_STATUS_SUCCESS {
                for d in descrs.iter().rev() {
                    let _ = (cudnn.backend_destroy_descriptor)(*d);
                }
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "cuDNN create tensor descriptor".into(),
                });
            }
            let dtype = cudnn_data_type(tensor.dtype).unwrap_or(CUDNN_DATA_FLOAT);
            let dims: Vec<i64> = tensor.shape.iter().map(|d| i64::try_from(*d).unwrap()).collect();
            let strides: Vec<i64> = (0..dims.len()).map(|i| dims[i + 1..].iter().product()).collect();
            let is_virtual = tensor.is_virtual as i32;
            if (cudnn.backend_set_attribute)(
                desc,
                CUDNN_ATTR_TENSOR_DATA_TYPE,
                CUDNN_TYPE_DATA_TYPE,
                1,
                (&raw const dtype).cast(),
            ) != CUDNN_STATUS_SUCCESS
                || (cudnn.backend_set_attribute)(
                    desc,
                    CUDNN_ATTR_TENSOR_DIMENSIONS,
                    CUDNN_TYPE_INT64,
                    dims.len() as i64,
                    dims.as_ptr().cast(),
                ) != CUDNN_STATUS_SUCCESS
                || (cudnn.backend_set_attribute)(
                    desc,
                    CUDNN_ATTR_TENSOR_STRIDES,
                    CUDNN_TYPE_INT64,
                    strides.len() as i64,
                    strides.as_ptr().cast(),
                ) != CUDNN_STATUS_SUCCESS
                || (cudnn.backend_set_attribute)(
                    desc,
                    CUDNN_ATTR_TENSOR_UNIQUE_ID,
                    CUDNN_TYPE_INT64,
                    1,
                    (&raw const tensor.uid).cast(),
                ) != CUDNN_STATUS_SUCCESS
                || (cudnn.backend_set_attribute)(
                    desc,
                    CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                    CUDNN_TYPE_BOOLEAN,
                    1,
                    (&raw const is_virtual).cast(),
                ) != CUDNN_STATUS_SUCCESS
                || (cudnn.backend_finalize)(desc) != CUDNN_STATUS_SUCCESS
            {
                for d in descrs.iter().rev() {
                    let _ = (cudnn.backend_destroy_descriptor)(*d);
                }
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "cuDNN configure tensor descriptor".into(),
                });
            }
            tensor_descs.push(desc);
            descrs.push(desc);
        }

        let mut op_descs: Vec<cudnnBackendDescriptor_t> = Vec::new();
        for op in &graph.ops {
            match unsafe { build_cudnn_op(cudnn, op, &tensor_descs, &graph.tensors, &mut descrs) } {
                Ok(op_desc) => {
                    op_descs.push(op_desc);
                }
                Err(e) => {
                    for d in descrs.iter().rev() {
                        let _ = (cudnn.backend_destroy_descriptor)(*d);
                    }
                    return Err(e);
                }
            }
        }

        // Operation graph.
        let mut opgraph = ptr::null_mut();
        if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &raw mut opgraph) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_set_attribute)(
                opgraph,
                CUDNN_ATTR_OPERATIONGRAPH_OPS,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                op_descs.len() as i64,
                op_descs.as_ptr().cast(),
            ) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_finalize)(opgraph) != CUDNN_STATUS_SUCCESS
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN operation graph".into() });
        }
        descrs.push(opgraph);

        // Engine heuristics: query the ranked list of engine configs.
        let mut heur = ptr::null_mut();
        let heur_mode = CUDNN_HEUR_MODE_INSTANT;
        if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &raw mut heur) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_set_attribute)(
                heur,
                CUDNN_ATTR_ENGINEHEUR_MODE,
                CUDNN_TYPE_HEUR_MODE,
                1,
                (&raw const heur_mode).cast(),
            ) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_set_attribute)(
                heur,
                CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                (&raw const opgraph).cast(),
            ) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_finalize)(heur) != CUDNN_STATUS_SUCCESS
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN engine heuristics".into() });
        }
        descrs.push(heur);

        let mut num_engines: i64 = 0;
        if (cudnn.backend_get_attribute)(
            heur,
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            0,
            &raw mut num_engines,
            ptr::null_mut(),
        ) != CUDNN_STATUS_SUCCESS
            || num_engines <= 0
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN heuristic engine query".into() });
        }
        let mut engine_configs: Vec<cudnnBackendDescriptor_t> = vec![ptr::null_mut(); num_engines as usize];
        if (cudnn.backend_get_attribute)(
            heur,
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            num_engines,
            &raw mut num_engines,
            engine_configs.as_mut_ptr().cast(),
        ) != CUDNN_STATUS_SUCCESS
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "cuDNN heuristic engine results".into(),
            });
        }
        for cfg in &engine_configs {
            descrs.push(*cfg);
        }

        // Execution plan from the best engine config.
        let mut plan = ptr::null_mut();
        if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &raw mut plan) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_set_attribute)(
                plan,
                CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                (&raw const engine_configs[0]).cast(),
            ) != CUDNN_STATUS_SUCCESS
            || (cudnn.backend_finalize)(plan) != CUDNN_STATUS_SUCCESS
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN execution plan".into() });
        }

        // Query the required workspace size and allocate it.
        let mut workspace_bytes: i64 = 0;
        if (cudnn.backend_get_attribute)(
            plan,
            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            CUDNN_TYPE_INT64,
            1,
            &raw mut workspace_bytes,
            (&raw mut workspace_bytes).cast(),
        ) != CUDNN_STATUS_SUCCESS
        {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN workspace query".into() });
        }
        let mut workspace: u64 = 0;
        if workspace_bytes > 0 && (cuMemAlloc)(&raw mut workspace, workspace_bytes as usize) != CUDAStatus::CUDA_SUCCESS {
            for d in descrs.iter().rev() {
                let _ = (cudnn.backend_destroy_descriptor)(*d);
            }
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "cuDNN workspace alloc".into() });
        }
        descrs.push(plan);

        Ok(CudnnPlan { plan, arg_uids: graph.arg_uids.clone(), workspace, workspace_bytes: workspace_bytes as Dim, descrs })
    }
}

/// Builds the cuDNN op descriptor for a single [`CudnnOp`]. All descriptors it
/// creates are appended to `descrs` so the caller can clean them up on error.
/// Returns the finalized op descriptor.
unsafe fn build_cudnn_op(
    cudnn: &CudnnLib,
    op: &CudnnOp,
    tensor_descs: &[cudnnBackendDescriptor_t],
    tensors: &[CudnnTensor],
    descrs: &mut Vec<cudnnBackendDescriptor_t>,
) -> Result<cudnnBackendDescriptor_t, BackendError> {
    unsafe {
        let uid_of = |uid: &i64| tensors.iter().position(|t| &t.uid == uid).map(|i| tensor_descs[i]).unwrap_or(ptr::null_mut());
        match op {
            CudnnOp::Matmul { a, b, c, compute_dtype } => {
                let a_desc = uid_of(a);
                let b_desc = uid_of(b);
                let c_desc = uid_of(c);
                let compute_dtype = cudnn_data_type(*compute_dtype).unwrap_or(CUDNN_DATA_FLOAT);

                // Matmul config descriptor: compute type.
                let mut mm_cfg = ptr::null_mut();
                if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &raw mut mm_cfg) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_set_attribute)(
                        mm_cfg,
                        CUDNN_ATTR_MATMUL_COMP_TYPE,
                        CUDNN_TYPE_DATA_TYPE,
                        1,
                        (&raw const compute_dtype).cast(),
                    ) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_finalize)(mm_cfg) != CUDNN_STATUS_SUCCESS
                {
                    return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "cuDNN matmul config".into() });
                }
                descrs.push(mm_cfg);

                let mut op_desc = ptr::null_mut();
                if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &raw mut op_desc)
                    != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_set_attribute)(
                        op_desc,
                        CUDNN_ATTR_OPERATION_MATMUL_ADESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR,
                        1,
                        (&raw const a_desc).cast(),
                    ) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_set_attribute)(
                        op_desc,
                        CUDNN_ATTR_OPERATION_MATMUL_BDESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR,
                        1,
                        (&raw const b_desc).cast(),
                    ) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_set_attribute)(
                        op_desc,
                        CUDNN_ATTR_OPERATION_MATMUL_CDESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR,
                        1,
                        (&raw const c_desc).cast(),
                    ) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_set_attribute)(
                        op_desc,
                        CUDNN_ATTR_OPERATION_MATMUL_DESC,
                        CUDNN_TYPE_BACKEND_DESCRIPTOR,
                        1,
                        (&raw const mm_cfg).cast(),
                    ) != CUDNN_STATUS_SUCCESS
                    || (cudnn.backend_finalize)(op_desc) != CUDNN_STATUS_SUCCESS
                {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "cuDNN matmul operation".into(),
                    });
                }
                descrs.push(op_desc);
                Ok(op_desc)
            }
        }
    }
}

/// Launches a compiled cuDNN plan on `stream`: builds a variant pack binding
/// the non-virtual tensor UIDs to the launch arg buffers (plus workspace) and
/// executes it.
#[allow(clippy::too_many_lines)]
unsafe fn launch_cudnn_plan(
    cudnn: &Option<Arc<CudnnLib>>,
    handle: Option<cudnnHandle_t>,
    plan: &CudnnPlan,
    buffers: &Slab<ChunkId, CUDABuffer>,
    args: &[LaunchArg],
    stream: CUstream,
) -> Result<(), BackendError> {
    unsafe {
        let Some(cudnn) = cudnn else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "cuDNN library not loaded.".into() });
        };
        let Some(handle) = handle else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "cuDNN handle missing.".into() });
        };
        let _ = stream;

        let mut variant_pack = ptr::null_mut();
        if (cudnn.backend_create_descriptor)(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &raw mut variant_pack) != CUDNN_STATUS_SUCCESS
        {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "cuDNN variant pack".into() });
        }

        let mut data_ptrs: Vec<*mut c_void> = Vec::with_capacity(args.len());
        for arg in args.iter() {
            match arg {
                LaunchArg::Variable(_) => todo!("scalar variant-pack entries in cuDNN launches"),
                LaunchArg::Buffer { chunk, offset, len } => {
                    data_ptrs.push(buffer_ptr(buffers, *chunk, *offset, *len)? as *mut c_void)
                }
            }
        }
        let unique_ids: Vec<i64> = plan.arg_uids.clone();

        let set_ok = (cudnn.backend_set_attribute)(
            variant_pack,
            CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
            CUDNN_TYPE_INT64,
            unique_ids.len() as i64,
            unique_ids.as_ptr().cast(),
        ) == CUDNN_STATUS_SUCCESS
            && (cudnn.backend_set_attribute)(
                variant_pack,
                CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                CUDNN_TYPE_VOID_PTR,
                data_ptrs.len() as i64,
                data_ptrs.as_ptr().cast(),
            ) == CUDNN_STATUS_SUCCESS
            && (if plan.workspace != 0 {
                let ws_ptr: *mut c_void = plan.workspace as *mut c_void;
                (cudnn.backend_set_attribute)(
                    variant_pack,
                    CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                    CUDNN_TYPE_VOID_PTR,
                    1,
                    (&raw const ws_ptr).cast(),
                ) == CUDNN_STATUS_SUCCESS
            } else {
                true
            })
            && (cudnn.backend_finalize)(variant_pack) == CUDNN_STATUS_SUCCESS;

        if set_ok {
            let _ = (cudnn.backend_execute)(handle, plan.plan, variant_pack);
        }
        let _ = (cudnn.backend_destroy_descriptor)(variant_pack);
        if set_ok {
            Ok(())
        } else {
            Err(BackendError { status: ErrorStatus::KernelLaunch, context: "cuDNN variant pack config".into() })
        }
    }
}

impl CUDADevice {
    fn get(
        &mut self,
        attr: CUdevice_attribute,
        cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUDAStatus,
    ) -> Result<c_int, BackendError> {
        let mut v = 0;
        unsafe { cuDeviceGetAttribute(&raw mut v, attr, self.device) }.check(ErrorStatus::DeviceQuery)?;
        Ok(v)
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
pub(super) struct CUmod_st {
    _unused: [u8; 0],
}
pub(super) type CUmodule = *mut CUmod_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(super) struct CUfunc_st {
    _unused: [u8; 0],
}
pub(super) type CUfunction = *mut CUfunc_st;
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUgraph_st {
    _unused: [u8; 0],
}
type CUgraph = *mut CUgraph_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUgraphExec_st {
    _unused: [u8; 0],
}
type CUgraphExec = *mut CUgraphExec_st;
/// Capture mode for `cuStreamBeginCapture` (driver value, global capture).
type CUstreamCaptureMode = c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUgraphNode_st {
    _unused: [u8; 0],
}
type CUgraphNode = *mut CUgraphNode_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUarray_st {
    _unused: [u8; 0],
}
type CUarray = *mut CUarray_st;
/// Memory-type tag for `CUDA_MEMCPY3D` (driver values).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum CUmemorytype {
    Host = 1,
    Device = 2,
    Array = 3,
    Unified = 4,
}
/// 3D memcpy descriptor: covers 1D/2D/3D node params (unused extents are 1).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUDA_MEMCPY3D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_z: usize,
    src_lod: usize,
    src_memory_type: CUmemorytype,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: CUarray,
    reserved0: *mut c_void,
    src_pitch: usize,
    src_height: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_z: usize,
    dst_lod: usize,
    dst_memory_type: CUmemorytype,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: CUarray,
    reserved1: *mut c_void,
    dst_pitch: usize,
    dst_height: usize,
    width_in_bytes: usize,
    height: usize,
    depth: usize,
}
/// Node-type tag for `cuGraphNodeGetType` (driver values; only the kernel
/// and memcpy variants are classified, everything else is rejected).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum CUgraphNodeType {
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Host = 3,
    Graph = 4,
    Empty = 5,
    WaitEvent = 6,
    EventRecord = 7,
    ExtSemasSignal = 8,
    ExtSemasWait = 9,
    BatchMemOp = 10,
    MemAlloc = 11,
    MemFree = 12,
}
/// Kernel-node params for capture-free node updates. v2 layout (CUDA 12+):
/// the trailing `kern`/`ctx` are only referenced when `func` is NULL —
/// `cuGraphExecUpdate` result info: only the status word is read here.
/// Full header layout (`CUgraphExecUpdateResultInfo_v1`); `result == 0`
/// is `CU_GRAPH_EXEC_UPDATE_SUCCESS`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUgraphExecUpdateResultInfo {
    result: u32,
    error_node: CUgraphNode,
    error_from_node: CUgraphNode,
}
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

#[repr(C)]
#[derive(Debug)]
struct _nvrtcProgram {
    _unused: [u8; 0],
}
type nvrtcProgram = *mut _nvrtcProgram;

#[allow(unused)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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
