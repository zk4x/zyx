// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
    dtype::{Constant, DType},
    error::{BackendError, ErrorStatus},
    graph::{ClassId, Graph},
    kernel::{BOp, Kernel, MMADims, Op, OpId, ParamKind, RangeKind, UOp},
    shape::Dim,
    slab::{Slab, SlabId},
};
use crate::{Map, hashers::FHasher};
use c::CDevice;
use cblas::CblasDevice;
use cuda::CUDADevice;
use dummy::DummyDevice;
use nanoserde::{DeBin, DeJson, SerBin};
use opencl::OpenCLDevice;
use std::sync::Mutex;
use std::{collections::BTreeSet, hash::BuildHasherDefault, sync::Arc};
#[cfg(feature = "tenstorrent")]
use tenstorrent::{TTDevice, TTMemoryPool};
use vulkan::VulkanDevice;
#[cfg(feature = "wgpu")]
use wgpu::{WGPUDevice, WGPUMemoryPool};

mod c;
mod cblas;
mod cuda;
mod disk;
mod dummy;
mod host;
mod opencl;
#[cfg(feature = "tenstorrent")]
mod tenstorrent;
mod vulkan;
#[cfg(feature = "wgpu")]
mod wgpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolBufferId(u32);

/// Global memory-pool handle. `Copy`, names both the backend and the ordinal,
/// and resolves directly to that pool's global `Arc<Mutex<...>>` — no slab ids.
///
/// Each variant owns its globals (one `Arc<Mutex<pool>>` per ordinal, singletons
/// for `Host`/`Disk`/`Dummy`), lazily initialized on first pool access. Pools are
/// process-wide: they outlive any `Runtime` and are never deinitialized.
/// The only lock takers are the device-API entry points below
/// (alloc/free/copy/compile/launch); the per-op tensor path never touches them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pool {
    /// Host RAM. Shared by the C and CBLAS devices, which own no pool.
    Host,
    /// Disk-backed tensors (paths, not bytes).
    Disk,
    /// CUDA device VRAM, one pool per GPU driver ordinal.
    Cuda(u16),
    /// OpenCL device memory, one pool per device.
    OpenCL(u16),
    /// Vulkan device memory, one pool per device.
    Vulkan(u16),
    /// Tenstorrent device memory, one pool per accelerator.
    #[cfg(feature = "tenstorrent")]
    TT(u16),
    /// WGPU device memory, one pool per device.
    #[cfg(feature = "wgpu")]
    WGPU(u16),
    /// Testing dummy pool (config-gated).
    Dummy,
}

pub(super) fn lock<'a, T>(pool: Pool, arc: &'a Arc<Mutex<T>>) -> std::sync::MutexGuard<'a, T> {
    arc.lock().unwrap_or_else(|_| panic!("{pool:?} pool lock poisoned by a panicking holder"))
}

impl Pool {
    /// All currently available pools. Ensures (and skips on failure) every
    /// backend table, so this triggers lazy init of all present hardware.
    /// Used for capacity queries like free-memory max.
    #[must_use]
    pub fn all() -> Vec<Pool> {
        let mut out = vec![Pool::Host, Pool::Disk];
        if dummy::pool().is_ok() {
            out.push(Pool::Dummy);
        }
        for i in 0..cuda::pool_count() {
            out.push(Pool::Cuda(i));
        }
        for i in 0..opencl::pool_count() {
            out.push(Pool::OpenCL(i));
        }
        for i in 0..vulkan::pool_count() {
            out.push(Pool::Vulkan(i));
        }
        #[cfg(feature = "tenstorrent")]
        for i in 0..tenstorrent::pool_count() {
            out.push(Pool::TT(i));
        }
        #[cfg(feature = "wgpu")]
        for i in 0..wgpu::pool_count() {
            out.push(Pool::WGPU(i));
        }
        out
    }

    /// Allocate a buffer. Lazily initializes the pool (and its device worker, if any).
    pub fn allocate(self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes = bytes + 8; // for the extra element, why not
        let free = self.free_bytes();
        let (result, name) = match self {
            Pool::Host => (lock(self, &host::pool()).allocate(bytes), "host"),
            Pool::Disk => todo!("disk is not allocatable"),
            Pool::Cuda(id) => (lock(self, &cuda::pool(id)?).allocate(bytes), "cuda"),
            Pool::OpenCL(id) => (lock(self, &opencl::pool(id)?).allocate(bytes), "opencl"),
            Pool::Vulkan(id) => (lock(self, &vulkan::pool(id)?).allocate(bytes), "vulkan"),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => (lock(self, &tenstorrent::pool(id)?).allocate(bytes), "tenstorrent"),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => (lock(self, &wgpu::pool(id)?).allocate(bytes), "wgpu"),
            Pool::Dummy => (lock(self, &dummy::pool()?).allocate(bytes), "dummy"),
        };
        if result.is_ok() {
            if let Ok(x) = std::env::var("ZYX_DEBUG")
                && let Ok(x) = x.parse::<u32>()
                && DebugMask::new(x).dev()
            {
                println!("[{name}] allocate {bytes} -> free {free} B");
            }
        } else {
            eprintln!("[{name}] allocate FAILED {bytes} -> free {free} B");
        }
        result
    }

    /// Free a buffer. The pool must already be initialized (a buffer id for it
    /// cannot exist otherwise); panics loudly if it is not.
    pub fn deallocate(self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let name = match self {
            Pool::Host => "host",
            Pool::Disk => "disk",
            Pool::Cuda(_) => "CUDA",
            Pool::OpenCL(_) => "OPENCL",
            Pool::Vulkan(_) => "Vulkan",
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => "tenstorrent",
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => "WGPU",
            Pool::Dummy => "dummy",
        };
        let free_before = self.free_bytes();
        match self {
            Pool::Host => lock(self, &host::pool()).deallocate(buffer_id, event_wait_list),
            Pool::Disk => lock(self, &disk::pool()).deallocate(buffer_id, event_wait_list),
            Pool::Cuda(id) => {
                lock(self, &cuda::pool(id).expect("deallocate on unavailable CUDA pool")).deallocate(buffer_id, event_wait_list)
            }
            Pool::OpenCL(id) => lock(self, &opencl::pool(id).expect("deallocate on unavailable OpenCL pool"))
                .deallocate(buffer_id, event_wait_list),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id).expect("deallocate on unavailable Vulkan pool"))
                .deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id).expect("deallocate on unavailable TT pool"))
                .deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                lock(self, &wgpu::pool(id).expect("deallocate on unavailable WGPU pool")).deallocate(buffer_id, event_wait_list)
            }
            Pool::Dummy => {
                lock(self, &dummy::pool().expect("deallocate on unavailable dummy pool")).deallocate(buffer_id, event_wait_list)
            }
        }
        if let Ok(x) = std::env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
            && DebugMask::new(x).memory()
        {
            let free_after = self.free_bytes();
            println!("[{name}] deallocate -> free {free_after} B (freed {} B)", free_after - free_before);
        }
    }

    pub fn free_bytes(self) -> Dim {
        match self {
            Pool::Host => lock(self, &host::pool()).free_bytes(),
            Pool::Disk => lock(self, &disk::pool()).free_bytes(),
            Pool::Cuda(id) => cuda::pool(id).map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
            Pool::OpenCL(id) => opencl::pool(id).map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
            Pool::Vulkan(id) => vulkan::pool(id).map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => tenstorrent::pool(id).map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => wgpu::pool(id).map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
            Pool::Dummy => dummy::pool().map(|p| lock(self, &p).free_bytes()).unwrap_or(0),
        }
    }

    pub fn host_to_pool(self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).host_to_pool(src, dst, event_wait_list),
            Pool::Disk => todo!("host to disk copy"),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).host_to_pool(src, dst, event_wait_list),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).host_to_pool(src, dst, event_wait_list),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => lock(self, &wgpu::pool(id)?).host_to_pool(src, dst, event_wait_list),
            Pool::Dummy => lock(self, &dummy::pool()?).host_to_pool(src, dst, event_wait_list),
        }
    }

    pub fn pool_to_host(self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).pool_to_host(src, dst, event_wait_list),
            Pool::Disk => lock(self, &disk::pool()).pool_to_host(src, dst, event_wait_list),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).pool_to_host(src, dst, event_wait_list),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).pool_to_host(src, dst, event_wait_list),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => lock(self, &wgpu::pool(id)?).pool_to_host(src, dst, event_wait_list),
            Pool::Dummy => lock(self, &dummy::pool()?).pool_to_host(src, dst, event_wait_list),
        }
    }

    /// Copy data from `src` pool into `self` (dst). Lock order is always
    /// dst-then-src; both are separate mutexes so this cannot deadlock as long
    /// as no path locks them in the opposite order.
    pub fn pool_to_pool(
        self,
        src: Pool,
        src_buf: PoolBufferId,
        dst_buf: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            Pool::Disk => todo!("copies into disk pool"),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => lock(self, &wgpu::pool(id)?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
            Pool::Dummy => lock(self, &dummy::pool()?).pool_to_pool(src, src_buf, dst_buf, event_wait_list),
        }
    }

    pub fn sync_events(self, events: Vec<Event>) -> Result<(), BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).sync_events(events),
            Pool::Disk => lock(self, &disk::pool()).sync_events(events),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).sync_events(events),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).sync_events(events),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).sync_events(events),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).sync_events(events),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => lock(self, &wgpu::pool(id)?).sync_events(events),
            Pool::Dummy => lock(self, &dummy::pool()?).sync_events(events),
        }
    }

    #[allow(unused)]
    pub fn release_events(self, events: Vec<Event>) {
        match self {
            Pool::Host => lock(self, &host::pool()).release_events(events),
            Pool::Disk => lock(self, &disk::pool()).release_events(events),
            Pool::Cuda(id) => {
                lock(self, &cuda::pool(id).expect("release_events on unavailable CUDA pool")).release_events(events)
            }
            Pool::OpenCL(id) => {
                lock(self, &opencl::pool(id).expect("release_events on unavailable OpenCL pool")).release_events(events)
            }
            Pool::Vulkan(id) => {
                lock(self, &vulkan::pool(id).expect("release_events on unavailable Vulkan pool")).release_events(events)
            }
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => {
                lock(self, &tenstorrent::pool(id).expect("release_events on unavailable TT pool")).release_events(events)
            }
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                lock(self, &wgpu::pool(id).expect("release_events on unavailable WGPU pool")).release_events(events)
            }
            Pool::Dummy => lock(self, &dummy::pool().expect("release_events on unavailable dummy pool")).release_events(events),
        }
    }
}

/// Read the backend config file (same search as `Runtime::initialize_backends`
/// used to do: `$XDG_CONFIG_HOME/zyx/config.json`, else `~/.config/zyx/config.json`).
/// Missing or unparsable file means defaults. Each pool/device lazy init reads
/// it independently; the file read is not on any hot path.
pub(crate) fn load_config() -> Config {
    use std::path::PathBuf;
    let debug = debug_backends();
    let config_file = std::env::var_os("XDG_CONFIG_HOME")
        .and_then(|path| {
            let path = PathBuf::from(path);
            if path.is_absolute() { Some(path) } else { None }
        })
        .or_else(|| std::env::home_dir().map(|home| home.join(".config")))
        .map(|path| path.join("zyx/config.json"))
        .and_then(|path| std::fs::read_to_string(&path).ok());
    config_file
        .and_then(|file| {
            DeJson::deserialize_json(&file)
                .map_err(|e| {
                    if debug {
                        println!("Failed to parse config.json, {e}");
                    }
                })
                .ok()
        })
        .inspect(|_| {
            if debug {
                println!("Device config successfully read and parsed.");
            }
        })
        .unwrap_or_else(|| {
            if debug {
                println!("Failed to get device config, using defaults.");
            }
            Config::default()
        })
}

/// Whether backend debug printing is enabled (`ZYX_DEBUG` device bit).
pub(crate) fn debug_backends() -> bool {
    std::env::var("ZYX_DEBUG").ok().and_then(|x| x.parse::<u32>().ok()).is_some_and(|x| DebugMask::new(x).dev())
}

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

/// Globally unique buffer identifier: the owning global pool plus the
/// buffer id within that pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    pub pool: Pool,
    pub buffer: PoolBufferId,
}

impl BufferId {
    pub const NULL: Self = Self { pool: Pool::Host, buffer: PoolBufferId(u32::MAX) };
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

pub fn initialize_backends(device_config: &Config, devices: &mut Slab<DeviceId, Device>, debug_backends: bool) {
    // Pools are process-wide globals now, resolved lazily via each backend's
    // `pool()` on first use. This only registers `Device` entries (stays until
    // the device-slab removal step).
    if let Err(err) = c::initialize_device(&device_config.c, devices, debug_backends)
        && debug_backends
    {
        println!("[C] {err}");
    }
    if let Err(err) = cblas::initialize_device(&device_config.cblas, devices, debug_backends)
        && debug_backends
    {
        println!("[cblas] {err}");
    }
    if let Err(err) = cuda::initialize_device(&device_config.cuda, devices, debug_backends)
        && debug_backends
    {
        println!("[cuda] {err}");
    }
    #[cfg(feature = "tenstorrent")]
    if let Err(err) = tenstorrent::initialize_device(&device_config.tenstorrent, devices, debug_backends) {
        if debug_backends {
            println!("[tenstorrent] {err}");
        }
    }
    if let Err(err) = vulkan::initialize_device(&device_config.vulkan, devices, debug_backends)
        && debug_backends
    {
        println!("[vulkan] {err}");
    }
    if let Err(err) = opencl::initialize_device(&device_config.opencl, devices, debug_backends)
        && debug_backends
    {
        println!("[opencl] {err}");
    }
    #[cfg(feature = "wgpu")]
    if let Err(err) = wgpu::initialize_device(&device_config.wgpu, devices, debug_backends)
        && debug_backends
    {
        println!("[wgpu] {err}");
    }
    if let Err(err) = dummy::initialize_device(&device_config.dummy, devices, debug_backends)
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
    /// Compute capability as [major, minor] (e.g. [7, 5] = sm_75).
    /// [0, 0] on devices without NVIDIA-style compute capability.
    pub cc: [i32; 2],
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
    /// Supported tile sizes [x, y] for tile-based accelerators (empty = no tile support)
    pub tile_sizes: Vec<[Dim; 2]>,
    /// Supported WMMA layouts (empty = no tensor core support)
    pub wmma_layouts: Vec<MMADims>,
    /// Number of hardware circular buffers (Tenstorrent: CB0-CB31 = 32).
    /// This is an architectural count and does not depend on the CB page
    /// size (2KB vs 4KB tiles); page size constrains L1 budget, not count.
    /// Zero on devices without circular buffers.
    pub num_circular_buffers: u32,
}

impl DeviceInfo {
    /// Returns operation capabilities for a dtype (none() if dtype is unsupported)
    pub const fn supports_dtype(&self, dtype: DType) -> DTypeCapability {
        self.dtype_capability[dtype as usize]
    }
}

#[derive(Debug)]
pub enum Device {
    C(CDevice),
    Cblas(CblasDevice),
    Dummy(DummyDevice),
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
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
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.deinitialize(),
            Device::Vulkan(dev) => dev.deinitialize(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.deinitialize(),
        }
    }

    pub fn info(&self) -> Arc<DeviceInfo> {
        match self {
            Device::C(dev) => dev.info(),
            Device::Cblas(dev) => dev.info(),
            Device::Dummy(dev) => dev.info(),
            Device::CUDA(dev) => dev.info(),
            Device::OpenCL(dev) => dev.info(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.info(),
            Device::Vulkan(dev) => dev.info(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.info(),
        }
    }

    pub const fn memory_pool(&self) -> Pool {
        match self {
            Device::C(dev) => dev.memory_pool(),
            Device::Cblas(dev) => dev.memory_pool(),
            Device::Dummy(dev) => dev.memory_pool(),
            Device::CUDA(dev) => dev.memory_pool(),
            Device::OpenCL(dev) => dev.memory_pool(),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.memory_pool(),
            Device::Vulkan(dev) => dev.memory_pool(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.memory_pool(),
        }
    }

    /// How much compute is available on the device,
    /// Internally this should be adjusted for current `device_usage`,
    /// so that we spread the laod across all available devices appropriatelly.
    pub fn free_compute(&self) -> u128 {
        match self {
            Device::C(dev) => dev.free_compute(),
            Device::Cblas(dev) => dev.free_compute(),
            Device::Dummy(dev) => dev.free_compute(),
            Device::CUDA(dev) => dev.free_compute(),
            Device::OpenCL(dev) => dev.free_compute(),
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
            #[cfg(feature = "tenstorrent")]
            Device::TT(_) => "Tenstorrent",
            Device::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            Device::WGPU(_) => "WGPU",
        }
    }

    /// CUDA compute capability, if available.
    #[cfg(feature = "viz")]
    pub fn compute_capability(&self) -> Option<[i32; 2]> {
        match self {
            Device::CUDA(dev) => Some(dev.compute_capability),
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
    /// parameter. `LaunchArg::Buffer` ids point into `pool` (which must be the
    /// device's own `memory_pool()`); `LaunchArg::Variable` carries the scalar
    /// value directly — backends never store variables. The grid (gws) is NOT
    /// passed here — each backend derives it at launch from the per-axis
    /// `GwsDim` it stored at compile, evaluating `Param(ordinal)` leaves
    /// against `args[ordinal]` (`LaunchArg::Variable` → `Constant::as_dim()`).
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        pool: Pool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        // A kernel always has at least one Param (its output); launching with
        // no args means buffer binding failed upstream — backends would pass
        // garbage param pointers to the driver.
        debug_assert!(!args.is_empty(), "launch with empty args: buffer binding failed upstream");
        match self {
            Device::C(dev) => dev.launch(program_id, pool, args, event_wait_list),
            Device::Cblas(dev) => dev.launch(program_id, pool, args, event_wait_list),
            Device::Dummy(dev) => dev.launch(program_id, pool, args, event_wait_list),
            Device::CUDA(dev) => dev.launch(program_id, pool, args, event_wait_list),
            Device::OpenCL(dev) => dev.launch(program_id, pool, args, event_wait_list),
            #[cfg(feature = "tenstorrent")]
            Device::TT(dev) => dev.launch(program_id, pool, args, event_wait_list),
            Device::Vulkan(dev) => dev.launch(program_id, pool, args, event_wait_list),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.launch(program_id, pool, args, event_wait_list),
        }
    }
}
