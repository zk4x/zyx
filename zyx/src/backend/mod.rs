// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues.
//! Interfaces use events independent from underlying implementation.
//! Events are used to achieve maximum asynchronous execution.

#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::upper_case_acronyms)]

// Because I don't want to write struct and inner enum for MemoryPool and Device

use crate::{
    DebugMask,
    backend::hip::{HIPDevice, HIPMemoryPool},
    dtype::{Constant, DType},
    error::{BackendError, ErrorStatus},
    graph::{ClassId, Graph},
    kernel::{BOp, Kernel, Op, OpId, ParamKind, RangeKind, UOp},
    shape::Dim,
    slab::{Slab, SlabId},
};
use crate::{Map, hashers::FHasher};
use c::CDevice;
use cblas::CblasDevice;
use cuda::{CUDADevice, CUDAMemoryPool};
use disk::DiskMemoryPool;
use dummy::{DummyDevice, DummyMemoryPool};
use host::HostMemoryPool;
use nanoserde::{DeBin, DeJson, SerBin};
use opencl::{OpenCLDevice, OpenCLMemoryPool};
use std::{collections::BTreeSet, hash::BuildHasherDefault};
#[cfg(feature = "tenstorrent")]
use tenstorrent::{TTDevice, TTMemoryPool};
use vulkan::{VulkanDevice, VulkanMemoryPool};
#[cfg(feature = "wgpu")]
use wgpu::{WGPUDevice, WGPUMemoryPool};

mod c;
mod cblas;
mod cuda;
mod disk;
mod dummy;
mod hip;
mod host;
mod opencl;
#[cfg(feature = "tenstorrent")]
mod tenstorrent;
mod vulkan;
#[cfg(feature = "wgpu")]
mod wgpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolBufferId(u32);

#[derive(Debug, Clone)]
pub enum LaunchArg {
    /// A plain data buffer (`PoolBufferId` indexes the launched `MemoryPool`).
    /// The caller guarantees every `Buffer` arg belongs to that pool.
    Buffer(PoolBufferId),
    /// A scalar value for a `Param { kind: Variable }`. Used both as a kernel
    /// param and (via group-index lengths) to derive the grid size host-side.
    Variable(Constant),
}

/// Per-gws-axis launch size for a compiled kernel. Backends store one of these
/// per gws axis at compile time and derive the actual grid at launch from it +
/// the bound `args`. See AGENTS.md "gws (Global Work Size)".
#[derive(Debug, Clone, PartialEq)]
pub enum GwsDim {
    /// The group length is an `Op::Const`; use this size directly.
    Const(Dim),
    /// The group length is an `Op::Param { kind: Variable }`; read
    /// `args[ordinal]`, which must be a `LaunchArg::Variable`, and take its
    /// value (`Constant::as_dim()`).
    Param(usize),
    /// The group length is a dim *expression* (e.g. an inferred reshape dim):
    /// a unary op over a recursively-evaluable group dim.
    Unary { x: Box<GwsDim>, uop: UOp },
    /// The group length is a dim *expression* (e.g. an inferred reshape dim):
    /// a binary op over two recursively-evaluable group dims.
    Binary { x: Box<GwsDim>, y: Box<GwsDim>, bop: BOp },
    /// The group length is a value-preserving cast of a dim expression (e.g.
    /// a `Variable` param cast to `IDX_T`).
    Cast { x: Box<GwsDim>, dtype: DType },
}

impl GwsDim {
    /// Evaluate to the concrete grid extent. `param` resolves a variable
    /// ordinal (same mapping as compile-time `Param` ordinals) to its value.
    #[must_use]
    pub fn eval(&self, param: &mut dyn FnMut(usize) -> Dim) -> Dim {
        match self {
            GwsDim::Const(d) => *d,
            GwsDim::Param(ordinal) => param(*ordinal),
            GwsDim::Unary { x, uop } => Constant::unary(Constant::idx(x.eval(param)), *uop)
                .as_dim()
                .expect("gws expression evaluated to a non-integer dim"),
            GwsDim::Binary { x, y, bop } => {
                let xv = Constant::idx(x.eval(param));
                let yv = Constant::idx(y.eval(param));
                Constant::binary(xv, yv, *bop).as_dim().expect("gws expression evaluated to a non-integer dim")
            }
            GwsDim::Cast { x, dtype } => {
                Constant::idx(x.eval(param)).cast(*dtype).as_dim().expect("cast gws expression evaluated to a non-integer dim")
            }
        }
    }
}

/// Walk the kernel's `Op::Index` ops and return one `GwsDim` per gws axis.
///
/// Each group length `op_id` is a dim expression over `Op::Const` and
/// `Op::Param { kind: Variable }` leaves, composed freely from
/// unary/binary/cast/load ops (`Const` → `Const`, `Param { Variable }` →
/// `Param`, and likewise for the composite ops); anything else is unreachable.
///
/// Constant group lengths are validated against `max_grid_dims` (the device's
/// per-axis max grid extent, `DeviceInfo::max_global_work_dims`); a constant
/// group length that exceeds the device limit is a compilation error. Symbolic
/// (param-backed) lengths cannot be checked here — backends must check the
/// evaluated grid extents before launching.
pub(crate) fn gws_from_kernel(kernel: &Kernel, max_grid_dims: &[Dim]) -> Result<Vec<GwsDim>, BackendError> {
    // Head-order position of every `Op::Param`, matching the arg ordering.
    let mut param_ordinal: Map<OpId, usize> = Map::with_hasher(BuildHasherDefault::<FHasher>::new());
    let mut param_idx = 0usize;
    let mut op_id = kernel.head;
    while !op_id.is_null() {
        if matches!(kernel.ops[op_id].op, Op::Param { .. }) {
            param_ordinal.insert(op_id, param_idx);
            param_idx += 1;
        }
        op_id = kernel.next_op(op_id);
    }

    fn conv(kernel: &Kernel, len: OpId, ordinals: &Map<OpId, usize>) -> GwsDim {
        match &kernel.ops[len].op {
            Op::Const(c) => GwsDim::Const(c.as_dim().unwrap()),
            Op::Param { kind: ParamKind::Variable, .. } => GwsDim::Param(ordinals[&len]),
            Op::Unary { x, uop } => GwsDim::Unary { x: Box::new(conv(kernel, *x, ordinals)), uop: *uop },
            Op::Binary { x, y, bop } => {
                GwsDim::Binary { x: Box::new(conv(kernel, *x, ordinals)), y: Box::new(conv(kernel, *y, ordinals)), bop: *bop }
            }
            // A load moves a value from global to local address space — it
            // never changes the value, so for length purposes it passes its
            // source through. Lengths only bottom out in `Param { Variable }`
            // leaves; a load from a buffer would be runtime data, not a dim.
            Op::Load { src, .. } => match &kernel.ops[*src].op {
                Op::Param { kind: ParamKind::Variable, .. } => GwsDim::Param(ordinals[src]),
                ref op => unreachable!("group length load from non-variable storage, got {op:?}"),
            },
            Op::Cast { x, dtype } => GwsDim::Cast { x: Box::new(conv(kernel, *x, ordinals)), dtype: *dtype },
            ref op => unreachable!("group length must be a dim over Const/Param Variable, got {op:?}"),
        }
    }

    let mut gws = Vec::new();
    let mut op_id = kernel.head;
    let mut steps_op_id = 0usize;
    while !op_id.is_null() {
        steps_op_id += 1;
        if steps_op_id > 10_000 {
            panic!("gws_from_kernel did not finish in 10000 steps");
        }
        if let Op::Range { axis, kind: RangeKind::Group(len) } = kernel.ops[op_id].op {
            let gdim = conv(kernel, len, &param_ordinal);
            let axis = axis as usize;
            if let GwsDim::Const(c) = gdim
                && let Some(&max) = max_grid_dims.get(axis)
                && c > max
            {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("grid dim {axis} {c} exceeds device max {max}").into(),
                });
            }
            if gws.len() <= axis {
                gws.resize(axis + 1, GwsDim::Const(1));
            }
            gws[axis] = gdim;
        }
        op_id = kernel.next_op(op_id);
    }
    Ok(gws)
}

impl From<usize> for PoolBufferId {
    fn from(value: usize) -> Self {
        PoolBufferId(u32::try_from(value).unwrap())
    }
}

impl From<PoolBufferId> for usize {
    fn from(value: PoolBufferId) -> Self {
        value.0 as usize
    }
}

impl SlabId for PoolBufferId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct DeviceProgramId(u32);

impl From<usize> for DeviceProgramId {
    fn from(value: usize) -> Self {
        DeviceProgramId(u32::try_from(value).unwrap())
    }
}

impl From<DeviceProgramId> for usize {
    fn from(value: DeviceProgramId) -> Self {
        value.0 as usize
    }
}

impl SlabId for DeviceProgramId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// Pool identifier for use with `Slab<PoolId, MemoryPool>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolId(u32);

impl PoolId {
    pub const HOST: Self = Self(0);
    pub const DISK: Self = Self(1);
}

impl From<usize> for PoolId {
    fn from(value: usize) -> Self {
        PoolId(u32::try_from(value).unwrap())
    }
}

impl From<PoolId> for usize {
    fn from(value: PoolId) -> Self {
        value.0 as usize
    }
}

impl SlabId for PoolId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl std::ops::AddAssign<u32> for PoolId {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

/// Device identifier for use with `Slab<DeviceId, Device>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct DeviceId(pub(crate) u32);

impl From<usize> for DeviceId {
    fn from(value: usize) -> Self {
        DeviceId(u32::try_from(value).unwrap())
    }
}

impl From<DeviceId> for usize {
    fn from(value: DeviceId) -> Self {
        value.0 as usize
    }
}

impl DeviceId {
    /// Auto-select the device (default scheduling behavior).
    pub const AUTO: Self = Self(u32::MAX);
}

impl SlabId for DeviceId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// Globally unique buffer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    pub pool: PoolId,
    pub buffer: PoolBufferId,
}

impl BufferId {
    pub const NULL: Self = Self { pool: PoolId::NULL, buffer: PoolBufferId(u32::MAX) };
}

impl From<usize> for BufferId {
    fn from(value: usize) -> Self {
        BufferId { pool: PoolId::ZERO, buffer: PoolBufferId(u32::try_from(value).unwrap()) }
    }
}

impl From<BufferId> for usize {
    fn from(value: BufferId) -> Self {
        value.buffer.0 as usize
    }
}

/// Globally unique program identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramId {
    pub device_id: DeviceId,
    pub program_id: DeviceProgramId,
}

impl ProgramId {
    pub const NULL: Self = Self { device_id: DeviceId::NULL, program_id: DeviceProgramId(u32::MAX) };
}

impl From<usize> for ProgramId {
    fn from(value: usize) -> Self {
        ProgramId { device_id: DeviceId::ZERO, program_id: DeviceProgramId(u32::try_from(value).unwrap()) }
    }
}

impl From<ProgramId> for usize {
    fn from(value: ProgramId) -> Self {
        value.program_id.0 as usize
    }
}

impl From<libloading::Error> for BackendError {
    fn from(value: libloading::Error) -> Self {
        BackendError { status: ErrorStatus::Initialization, context: value.to_string().into() }
    }
}

pub fn initialize_backends(
    device_config: &Config,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_backends: bool,
) {
    if let Err(err) = host::initialize_pool(memory_pools, debug_backends)
        && debug_backends
    {
        println!("[host] {err}");
    }
    if let Err(err) = disk::initialize_pool(memory_pools, debug_backends)
        && debug_backends
    {
        println!("[host] {err}");
    }
    if let Err(err) = c::initialize_device(&device_config.c, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[C] {err}");
    }
    if let Err(err) = cblas::initialize_device(&device_config.cblas, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[cblas] {err}");
    }
    if let Err(err) = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[cuda] {err}");
    }
    if let Err(err) = hip::initialize_device(&device_config.hip, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[HIP] {err}");
    }
    #[cfg(feature = "tenstorrent")]
    if let Err(err) = tenstorrent::initialize_device(&device_config.tenstorrent, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("[tenstorrent] {err}");
        }
    }
    if let Err(err) = vulkan::initialize_device(&device_config.vulkan, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[vulkan] {err}");
    }
    if let Err(err) = opencl::initialize_device(&device_config.opencl, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[opencl] {err}");
    }
    #[cfg(feature = "wgpu")]
    if let Err(err) = wgpu::initialize_device(&device_config.wgpu, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[wgpu] {err}");
    }
    if let Err(err) = dummy::initialize_device(&device_config.dummy, memory_pools, devices, debug_backends)
        && debug_backends
    {
        println!("[dummy] {err}");
    }
    //println!("YO {:?}", devices[DeviceId::from(0)].info().supported_dtypes);

    if devices.is_empty() {
        println!("All devices failed to initialize or were configured out.");
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum Event {
    #[allow(unused)]
    Disk(disk::DiskEvent),
    Host(host::HostEvent),
    CUDA(cuda::CUDAEvent),
    OpenCL(opencl::OpenCLEvent),
    HIP(hip::HIPEvent),
    #[cfg(feature = "tenstorrent")]
    TT(tenstorrent::TTEvent),
    Vulkan(vulkan::VulkanEvent),
    #[cfg(feature = "wgpu")]
    WGPU(wgpu::WGPUEvent),
}

/// Device configuration
#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(DeJson, Debug, Default)]
#[nserde(default)]
pub struct Config {
    /// Kernel autotune configuration
    pub autotune: crate::kernel::autotune::BeamSearch,
    /// C/Clang backend configuration
    pub c: c::CConfig,
    /// CBLAS backend configuration
    pub cblas: cblas::CblasConfig,
    /// Configuration of dummy device for testing
    pub dummy: dummy::DummyConfig,
    /// CUDA configuration
    pub cuda: cuda::CUDAConfig,
    /// HIP configuration
    pub hip: hip::HIPConfig,
    /// `OpenCL` configuration
    pub opencl: opencl::OpenCLConfig,
    /// Tenstorrent configuration
    #[cfg(feature = "tenstorrent")]
    pub tenstorrent: tenstorrent::TTConfig,
    // Vulkan configuration
    pub vulkan: vulkan::VulkanConfig,
    /// WGSL configuration
    #[cfg(feature = "wgpu")]
    pub wgpu: wgpu::WGPUConfig,
}

/// Per-dtype capability bitmask — one bit per unary/binary operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct DTypeCapability(u32);

impl DTypeCapability {
    pub const ZERO: Self = Self(0);
}

macro_rules! op_cap {
    ($name:ident, $bit:expr, $method:ident) => {
        pub const $name: Self = Self(1 << $bit);
        pub fn $method(&self) -> bool {
            self.0 & Self::$name.0 != 0
        }
    };
}

impl std::ops::BitOr for DTypeCapability {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl DTypeCapability {
    op_cap!(NEG, 0, neg);
    op_cap!(BITNOT, 1, bitnot);
    op_cap!(EXP, 2, exp);
    op_cap!(EXP2, 3, exp2);
    op_cap!(LN, 4, ln);
    op_cap!(LOG2, 5, log2);
    op_cap!(RECIPROCAL, 6, reciprocal);
    op_cap!(SQRT, 7, sqrt);
    op_cap!(SIN, 8, sin);
    op_cap!(COS, 9, cos);
    op_cap!(FLOOR, 10, floor);
    op_cap!(TRUNC, 11, trunc);
    op_cap!(ABS, 12, abs);
    op_cap!(ADD, 13, add);
    op_cap!(SUB, 14, sub);
    op_cap!(MUL, 15, mul);
    op_cap!(DIV, 16, div);
    op_cap!(POW, 17, pow);
    op_cap!(MOD, 18, r#mod);
    op_cap!(CMPLT, 19, cmplt);
    op_cap!(CMPGT, 20, cmpgt);
    op_cap!(MAX, 21, max);
    op_cap!(OR, 22, or);
    op_cap!(AND, 23, and);
    op_cap!(BITXOR, 24, bitxor);
    op_cap!(BITOR, 25, bitor);
    op_cap!(BITAND, 26, bitand);
    op_cap!(BITSHIFTLEFT, 27, bitshiftleft);
    op_cap!(BITSHIFTRIGHT, 28, bitshiftright);
    op_cap!(NOTEQ, 29, noteq);
    op_cap!(EQ, 30, eq);

    #[must_use]
    pub const fn all() -> Self {
        Self(u32::MAX)
    }

    #[must_use]
    pub const fn none() -> Self {
        Self(0)
    }

    #[must_use]
    pub fn any(&self) -> bool {
        self.0 != 0
    }

    #[must_use]
    pub fn invert(&self) -> Self {
        Self(!self.0)
    }

    #[must_use]
    pub fn exclude(&self, capability: DTypeCapability) -> Self {
        Self(self.0 & !capability.0)
    }

    #[must_use]
    pub fn include(&self, capability: DTypeCapability) -> Self {
        Self(self.0 | capability.0)
    }
}

/// Hardware information needed for applying optimizations
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
    /// Biggest kernel dimensions
    pub max_global_work_dims: Vec<Dim>,
    /// Maximum local work size threads
    pub max_local_threads: u32,
    /// Maximum local work size dimensions
    pub max_local_work_dims: Vec<u32>,
    /// Preferred vector size in bytes
    pub preferred_vector_size: u8,
    /// Local memory size in bytes
    pub local_mem_size: Dim,
    /// private memory size in bytes
    pub max_register_bytes: Dim,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
    /// Warp size
    pub warp_size: u16,
    /// Per-dtype operation capabilities
    pub dtype_capability: [DTypeCapability; DType::N_DTYPES],
    /// Whether the device has a native exp2 instruction
    pub has_native_exp2: bool,
    /// Supported vector lengths for loads/stores/compute
    pub supported_vec_lens: Vec<u8>,
    /// Whether this device is a Tenstorrent Tensix accelerator
    pub tenstorrent: bool,
    /// Native tile shape [x, y] for tile-based (SIMD) accelerators
    pub tile: [Dim; 2],
}

impl DeviceInfo {
    /// Returns operation capabilities for a dtype (none() if dtype is unsupported)
    pub const fn supports_dtype(&self, dtype: DType) -> DTypeCapability {
        self.dtype_capability[dtype as usize]
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum MemoryPool {
    Dummy(DummyMemoryPool),
    Disk(DiskMemoryPool),
    Host(HostMemoryPool),
    CUDA(CUDAMemoryPool),
    OpenCL(OpenCLMemoryPool),
    HIP(HIPMemoryPool),
    #[cfg(feature = "tenstorrent")]
    TT(TTMemoryPool),
    Vulkan(VulkanMemoryPool),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUMemoryPool),
}

impl MemoryPool {
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        match self {
            MemoryPool::Dummy(pool) => pool.deinitialize(),
            MemoryPool::Disk(pool) => pool.deinitialize(),
            MemoryPool::Host(pool) => pool.deinitialize(),
            MemoryPool::CUDA(pool) => pool.deinitialize(),
            MemoryPool::OpenCL(pool) => pool.deinitialize(),
            MemoryPool::HIP(pool) => pool.deinitialize(),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.deinitialize(),
            MemoryPool::Vulkan(pool) => pool.deinitialize(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deinitialize(),
        }
    }

    pub const fn disk_pool(&mut self) -> Option<&mut DiskMemoryPool> {
        match self {
            Self::Disk(disk) => Some(disk),
            _ => None,
        }
    }

    pub fn free_bytes(&self) -> Dim {
        match self {
            MemoryPool::Dummy(pool) => pool.free_bytes(),
            MemoryPool::Disk(pool) => pool.free_bytes(),
            MemoryPool::Host(pool) => pool.free_bytes(),
            MemoryPool::CUDA(pool) => pool.free_bytes(),
            MemoryPool::OpenCL(pool) => pool.free_bytes(),
            MemoryPool::HIP(pool) => pool.free_bytes(),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.free_bytes(),
            MemoryPool::Vulkan(pool) => pool.free_bytes(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.free_bytes(),
        }
    }

    /// Allocate a buffer. Returns (buffer_id, event) where the event signals
    /// when the buffer is ready for use. For most backends the event is a no-op
    /// (immediately signaled); CUDA returns an event recorded after the async allocation.
    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes = bytes + 8; // for the extra element, why not
        let free = self.free_bytes();
        let (result, name) = match self {
            MemoryPool::Dummy(pool) => (pool.allocate(bytes), "dummy"),
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::Host(pool) => (pool.allocate(bytes), "host"),
            MemoryPool::CUDA(pool) => (pool.allocate(bytes), "cuda"),
            MemoryPool::OpenCL(pool) => (pool.allocate(bytes), "opencl"),
            MemoryPool::HIP(pool) => (pool.allocate(bytes), "hip"),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => (pool.allocate(bytes), "tenstorrent"),
            MemoryPool::Vulkan(pool) => (pool.allocate(bytes), "vulkan"),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => (pool.allocate(bytes), "wgpu"),
        };
        if result.is_ok() {
            if let Ok(x) = std::env::var("ZYX_DEBUG")
                && let Ok(x) = x.parse::<u32>()
                && DebugMask(x).memory()
            {
                println!("[{name}] allocate {bytes} -> free {free} B");
            }
        } else {
            eprintln!("[{name}] allocate FAILED {bytes} -> free {free} B");
        }
        result
    }

    /// Free a buffer. Waits on all events in `event_wait_list` before freeing
    /// the underlying device memory. The events are consumed (waited and dropped)
    /// so callers must not reuse them.
    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let name = match self {
            MemoryPool::Dummy(_) => "dummy",
            MemoryPool::Disk(_) => "disk",
            MemoryPool::Host(_) => "host",
            MemoryPool::CUDA(_) => "CUDA",
            MemoryPool::OpenCL(_) => "OPENCL",
            MemoryPool::HIP(_) => "HIP",
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(_) => "tenstorrent",
            MemoryPool::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(_) => "WGPU",
        };
        let free_before = self.free_bytes();
        match self {
            MemoryPool::Dummy(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Disk(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Host(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::CUDA(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::HIP(pool) => pool.deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Vulkan(pool) => pool.deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deallocate(buffer_id, event_wait_list),
        }
        if let Ok(x) = std::env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
            && DebugMask(x).memory()
        {
            let free_after = self.free_bytes();
            println!("[{name}] deallocate -> free {free_after} B (freed {} B)", free_after - free_before);
        }
    }

    /// Copy data from host memory to a device buffer. Waits on all events in
    /// `event_wait_list` before starting the copy. Returns an event that signals
    /// when the copy is complete. The input events are consumed (waited and dropped).
    ///
    /// Backends that use synchronous copies (e.g. Vulkan with HOST_COHERENT memory)
    /// return a no-op event that is already signaled.
    pub fn host_to_pool(
        &mut self,
        src: &[u8], // TODO this will likely have to be Vec<u8> for better lifetimes handling and less synchronization
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::Host(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::CUDA(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::HIP(pool) => pool.host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::Vulkan(pool) => pool.host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.host_to_pool(src, dst, event_wait_list),
        }
    }

    /// Copy data from a device buffer to host memory. Waits on all events in
    /// `event_wait_list` before starting the copy (ensures GPU writes are visible).
    /// Blocking — does not return until the copy completes. The input events are
    /// consumed (waited and dropped).
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Disk(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Host(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::CUDA(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::HIP(pool) => pool.pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Vulkan(pool) => pool.pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.pool_to_host(src, dst, event_wait_list),
        }
    }

    /// Copy data from src to dst pool
    pub fn pool_to_pool(
        &mut self,
        src_pool: &mut MemoryPool,
        src: PoolBufferId,
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let event = match self {
            MemoryPool::Dummy(_) => todo!(),
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::Host(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
            MemoryPool::CUDA(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
            MemoryPool::OpenCL(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
            MemoryPool::HIP(_) => todo!(),
            MemoryPool::Vulkan(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(memory_pool) => memory_pool.pool_to_pool(src_pool, src, dst, event_wait_list)?,
        };
        Ok(event)
    }

    /// Wait for GPU events to complete, then drop them. Blocking.
    /// Used after host_to_pool or test-launch to ensure data is fully transferred.
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.sync_events(events),
            MemoryPool::Disk(pool) => pool.sync_events(events),
            MemoryPool::Host(pool) => pool.sync_events(events),
            MemoryPool::CUDA(pool) => pool.sync_events(events),
            MemoryPool::OpenCL(pool) => pool.sync_events(events),
            MemoryPool::HIP(pool) => pool.sync_events(events),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.sync_events(events),
            MemoryPool::Vulkan(pool) => pool.sync_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.sync_events(events),
        }
    }

    /// Drop events without waiting for GPU completion. Non-blocking.
    /// Used for cleanup when the graph is done and events are no longer needed
    /// (the final pool_to_host already waited for all GPU work).
    #[allow(unused)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        match self {
            MemoryPool::Dummy(pool) => pool.release_events(events),
            MemoryPool::Disk(pool) => pool.release_events(events),
            MemoryPool::Host(pool) => pool.release_events(events),
            MemoryPool::CUDA(pool) => pool.release_events(events),
            MemoryPool::OpenCL(pool) => pool.release_events(events),
            MemoryPool::HIP(pool) => pool.release_events(events),
            #[cfg(feature = "tenstorrent")]
            MemoryPool::TT(pool) => pool.release_events(events),
            MemoryPool::Vulkan(pool) => pool.release_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.release_events(events),
        }
    }
}

#[derive(Debug)]
pub enum Device {
    C(CDevice),
    Cblas(CblasDevice),
    Dummy(DummyDevice),
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
    HIP(HIPDevice),
    #[cfg(feature = "tenstorrent")]
    TT(TTDevice),
    Vulkan(VulkanDevice),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUDevice),
}

impl Device {
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        match self {
            Device::C(dev) => dev.deinitialize(),
            Device::Cblas(dev) => dev.deinitialize(),
            Device::Dummy(dev) => dev.deinitialize(),
            Device::CUDA(dev) => dev.deinitialize(),
            Device::OpenCL(dev) => dev.deinitialize(),
            Device::HIP(dev) => dev.deinitialize(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.deinitialize(),
            Device::Vulkan(dev) => dev.deinitialize(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.deinitialize(),
        }
    }

    pub const fn info(&self) -> &DeviceInfo {
        match self {
            Device::C(dev) => dev.info(),
            Device::Cblas(dev) => dev.info(),
            Device::Dummy(dev) => dev.info(),
            Device::CUDA(dev) => dev.info(),
            Device::OpenCL(dev) => dev.info(),
            Device::HIP(dev) => dev.info(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.info(),
            Device::Vulkan(dev) => dev.info(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.info(),
        }
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        match self {
            Device::C(dev) => dev.memory_pool_id(),
            Device::Cblas(dev) => dev.memory_pool_id(),
            Device::Dummy(dev) => dev.memory_pool_id(),
            Device::CUDA(dev) => dev.memory_pool_id(),
            Device::OpenCL(dev) => dev.memory_pool_id(),
            Device::HIP(dev) => dev.memory_pool_id(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.memory_pool_id(),
            Device::Vulkan(dev) => dev.memory_pool_id(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.memory_pool_id(),
        }
    }

    /// How much compute is available on the device,
    /// Internally this should be adjusted for current `device_usage`,
    /// so that we spread the laod across all available devices appropriatelly.
    pub const fn free_compute(&self) -> u128 {
        match self {
            Device::C(dev) => dev.free_compute(),
            Device::Cblas(dev) => dev.free_compute(),
            Device::Dummy(dev) => dev.free_compute(),
            Device::CUDA(dev) => dev.free_compute(),
            Device::OpenCL(dev) => dev.free_compute(),
            Device::HIP(dev) => dev.free_compute(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.free_compute(),
            Device::Vulkan(dev) => dev.free_compute(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.free_compute(),
        }
    }

    /// Whether this device only runs AOT (precompiled) kernels and cannot
    /// compile generic zyx kernels (e.g. the cblas backend). Such devices
    /// must be skipped by generic kernel autotuning.
    pub const fn aot_only(&self) -> bool {
        matches!(self, Self::Cblas(_))
    }

    /// Human-readable device name (e.g. "CUDA", "OpenCL", "C").
    #[cfg(feature = "viz")]
    pub const fn name(&self) -> &'static str {
        match self {
            Device::C(_) => "C",
            Device::Cblas(_) => "CBLAS",
            Device::Dummy(_) => "Dummy",
            Device::CUDA(_) => "CUDA",
            Device::OpenCL(_) => "OpenCL",
            Device::HIP(_) => "HIP",
            #[cfg(feature = "tenstorrent")]
            Device::TT(_) => "Tenstorrent",
            Device::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            Device::WGPU(_) => "WGPU",
        }
    }

    /// CUDA/HIP compute capability, if available.
    #[cfg(feature = "viz")]
    pub fn compute_capability(&self) -> Option<[i32; 2]> {
        match self {
            Device::CUDA(dev) => Some(dev.compute_capability),
            Device::HIP(dev) => Some(dev.compute_capability),
            _ => None,
        }
    }

    /// Whether the C backend was compiled with OpenMP support.
    #[cfg(feature = "viz")]
    pub fn has_openmp(&self) -> bool {
        match self {
            Device::C(dev) => dev.has_openmp,
            _ => false,
        }
    }

    /// Compile a kernel into a device program. Returns a program ID usable with
    /// `launch` and `release`. The `debug_asm` flag controls whether the backend
    /// prints the compiled assembly/source (for `ZYX_DEBUG=16`).
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let name = match self {
            Device::C(_) => "C",
            Device::Cblas(_) => "cblas",
            Device::Dummy(_) => "dummy",
            Device::CUDA(_) => "CUDA",
            Device::OpenCL(_) => "OPENCL",
            Device::HIP(_) => "HIP",
            #[cfg(feature = "tenstorrent")]
            Device::TT(_) => "tenstorrent",
            Device::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            Device::WGPU(_) => "WGPU",
        };
        let result = match self {
            Device::C(dev) => dev.compile(kernel, debug_asm),
            Device::Cblas(dev) => dev.compile(kernel, debug_asm),
            Device::Dummy(dev) => dev.compile(kernel, debug_asm),
            Device::CUDA(dev) => dev.compile(kernel, debug_asm),
            Device::OpenCL(dev) => dev.compile(kernel, debug_asm),
            Device::HIP(dev) => dev.compile(kernel, debug_asm),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.compile(kernel, debug_asm),
            Device::Vulkan(dev) => dev.compile(kernel, debug_asm),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.compile(kernel, debug_asm),
        };
        if let Ok(x) = std::env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
            && DebugMask(x).compile()
        {
            println!("[{name}] compile kernel");
        }
        result
    }

    /// Free a compiled program and its device resources (pipeline, shader module, etc.).
    pub fn release(&mut self, program_id: DeviceProgramId) {
        match self {
            Device::C(dev) => dev.release(program_id),
            Device::Cblas(dev) => dev.release(program_id),
            Device::Dummy(dev) => dev.release(program_id),
            Device::CUDA(dev) => dev.release(program_id),
            Device::OpenCL(dev) => dev.release(program_id),
            Device::HIP(dev) => dev.release(program_id),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.release(program_id),
            Device::Vulkan(dev) => dev.release(program_id),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.release(program_id),
        }
    }

    /// Pattern-matches subgraphs in `graph` (e.g. matmul) and adds `Node::Kernel`s
    /// backed by this device's AOT kernels so they compete with the fused zyx
    /// kernels in extraction. No-op for devices without AOT kernels.
    pub fn match_graph(&mut self, graph: &mut Graph, outputs: &BTreeSet<ClassId>) {
        match self {
            Device::Cblas(dev) => dev.match_graph(graph, outputs),
            Device::CUDA(dev) => dev.match_graph(graph, outputs),
            _ => {}
        }
        // A vendor pass adds Node::Kernel nodes with input edges; those must
        // never close a dependency cycle over the class graph.
        graph.verify();
    }

    /// Launch a kernel on the device. Waits on all events in `event_wait_list`
    /// before submitting to the GPU queue (ensures input buffers are ready).
    /// Returns an event that signals when the kernel completes.
    ///
    /// The `args` are the `LaunchArg`s for the kernel in the order the
    /// `Param` ops appear in the kernel IR given to compile (flat, head order, all
    /// kinds: `Variable`/`Global`/`GlobalMut`). `Op::Storage` is NOT a kernel
    /// parameter. `LaunchArg::Buffer` ids point into `memory_pool`;
    /// `LaunchArg::Variable` carries the scalar value directly — backends never
    /// store variables. The grid (gws) is NOT passed here — each backend derives
    /// it at launch from the per-axis `GwsDim` it stored at compile, evaluating
    /// `Param(ordinal)` leaves against `args[ordinal]` (`LaunchArg::Variable`
    /// → `Constant::as_dim()`).
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut MemoryPool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        // A kernel always has at least one Param (its output); launching with
        // no args means buffer binding failed upstream — backends would pass
        // garbage param pointers to the driver.
        debug_assert!(!args.is_empty(), "launch with empty args: buffer binding failed upstream");
        match self {
            Device::C(dev) => {
                let MemoryPool::Host(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::Cblas(dev) => {
                let MemoryPool::Host(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::Dummy(dev) => {
                let MemoryPool::Dummy(pool) = memory_pool else {
                    unreachable!()
                };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::CUDA(dev) => {
                let MemoryPool::CUDA(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::OpenCL(dev) => {
                let MemoryPool::OpenCL(pool) = memory_pool else {
                    unreachable!()
                };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::HIP(dev) => {
                let MemoryPool::HIP(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => {
                let MemoryPool::TT(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::Vulkan(dev) => {
                let MemoryPool::Vulkan(pool) = memory_pool else {
                    unreachable!()
                };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => {
                let MemoryPool::WGPU(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
        }
    }
}
