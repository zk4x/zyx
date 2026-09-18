// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues.
//! Interfaces use events independent from underlying implementation.
//! Events are used to achieve maximum asynchronous execution.
//!
//! Handles and ownership:
//! - [`Pool`] and [`Dev`] are `Copy` handles. They name the backend plus the
//!   ordinal and resolve directly to that backend's process-wide globals —
//!   one `Arc<Mutex<pool>>` per pool ordinal, one `Arc<Mutex<device>>` per
//!   device. There are no tables, no slab ids, no `Runtime`-owned pools/devices.
//! - The mapping direction is fixed: **the pool is always derived from the
//!   device, never the other way around** ([`Dev::pool`] is a pure function).
//!   Nothing derives a device from a pool; callers that need both take the
//!   `Dev` and derive the `Pool` from it.
//! - Both pools and devices are lazily initialized on first access, per
//!   backend, from the backend config file (see [`config`]). The only
//!   lock takers are the device-API entry points (alloc/free/copy/compile/
//!   launch); the per-op tensor path never touches the globals directly.

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
    slab::SlabId,
};
use crate::{Map, hashers::FHasher};
use nanoserde::{DeBin, DeJson, SerBin};
use std::sync::Mutex;
use std::{collections::BTreeSet, hash::BuildHasherDefault, sync::Arc};

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

/// Device selector and handle. `Copy`, names the backend plus the hardware
/// ordinal, and resolves directly to that device's global `Arc<Mutex<...>>`.
///
/// `Auto` is the default scheduling selector (first available device);
/// every other variant is a concrete device. The device's memory pool is
/// always derived from the device via [`Dev::pool`] — never the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dev {
    /// Auto-select: resolves to the first available device from [`Dev::all`].
    Auto,
    /// CPU backend (runs on [`Pool::Host`]).
    C,
    /// CBLAS backend for AOT matmuls (runs on [`Pool::Host`]).
    Cblas,
    /// CUDA GPU with the given driver ordinal.
    Cuda(u16),
    /// Tenstorrent chip with the given id.
    #[cfg(feature = "tenstorrent")]
    TT(u16),
    /// Vulkan physical device with the given index.
    Vulkan(u16),
    /// OpenCL device with the given index.
    OpenCL(u16),
    /// WGPU device with the given index.
    #[cfg(feature = "wgpu")]
    WGPU(u16),
    /// Testing dummy device (config-gated).
    Dummy,
}

pub(super) fn lock<'a, T>(pool: Pool, arc: &'a Arc<Mutex<T>>) -> std::sync::MutexGuard<'a, T> {
    arc.lock().unwrap_or_else(|_| panic!("{pool:?} pool lock poisoned by a panicking holder"))
}

pub(super) fn dlock<'a, T>(dev: Dev, arc: &'a Arc<Mutex<T>>) -> std::sync::MutexGuard<'a, T> {
    arc.lock().unwrap_or_else(|_| panic!("{dev:?} device lock poisoned by a panicking holder"))
}

impl Dev {
    /// All currently available devices, triggering lazy init of every
    /// backend (backends that are configured out or whose hardware/driver
    /// is missing contribute nothing).
    #[must_use]
    pub fn all() -> Vec<Dev> {
        let mut out = Vec::new();
        if c::device().is_ok() {
            out.push(Dev::C);
        }
        if cblas::device().is_ok() {
            out.push(Dev::Cblas);
        }
        for i in 0..cuda::device_count() {
            out.push(Dev::Cuda(i));
        }
        #[cfg(feature = "tenstorrent")]
        for i in 0..tenstorrent::device_count() {
            out.push(Dev::TT(i));
        }
        for i in 0..vulkan::device_count() {
            out.push(Dev::Vulkan(i));
        }
        for i in 0..opencl::device_count() {
            out.push(Dev::OpenCL(i));
        }
        #[cfg(feature = "wgpu")]
        for i in 0..wgpu::device_count() {
            out.push(Dev::WGPU(i));
        }
        if dummy::device().is_ok() {
            out.push(Dev::Dummy);
        }
        out
    }

    /// The memory pool belonging to this device. Pure function — the pool
    /// is always derived from the device, never the reverse.
    #[must_use]
    pub fn pool(self) -> Pool {
        match self {
            Dev::Auto => panic!("Dev::Auto has no pool; resolve it with Dev::auto() first"),
            Dev::C | Dev::Cblas => Pool::Host,
            Dev::Cuda(i) => Pool::Cuda(i),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(i) => Pool::TT(i),
            Dev::Vulkan(i) => Pool::Vulkan(i),
            Dev::OpenCL(i) => Pool::OpenCL(i),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(i) => Pool::WGPU(i),
            Dev::Dummy => Pool::Dummy,
        }
    }

    /// Device info for this device.
    pub fn info(self) -> Arc<DeviceInfo> {
        match self {
            Dev::Auto => panic!("Dev::Auto has no info; resolve it with Dev::auto() first"),
            Dev::C => c::device().expect("C device unavailable").lock().unwrap().info(),
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().info(),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).info(),
            Dev::OpenCL(id) => dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).info(),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => dlock(self, &tenstorrent::device(id).expect("TT device unavailable")).info(),
            Dev::Vulkan(id) => dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).info(),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).info(),
            Dev::Dummy => dummy::device().expect("dummy device unavailable").lock().unwrap().info(),
        }
    }

    /// How much compute is available on the device.
    pub fn free_compute(self) -> u128 {
        match self {
            Dev::Auto => panic!("Dev::Auto has no compute; resolve it with Dev::auto() first"),
            Dev::C => c::device().expect("C device unavailable").lock().unwrap().free_compute(),
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().free_compute(),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).free_compute(),
            Dev::OpenCL(id) => dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).free_compute(),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => dlock(self, &tenstorrent::device(id).expect("TT device unavailable")).free_compute(),
            Dev::Vulkan(id) => dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).free_compute(),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).free_compute(),
            Dev::Dummy => dummy::device().expect("dummy device unavailable").lock().unwrap().free_compute(),
        }
    }

    // TODO remove this somehow perhaps
    /// Whether this device only runs AOT (precompiled) kernels and cannot
    /// compile generic zyx kernels (e.g. the cblas backend). Such devices
    /// must be skipped by generic kernel autotuning.
    #[must_use]
    pub const fn aot_only(self) -> bool {
        matches!(self, Self::Cblas)
    }

    /// Human-readable device name (e.g. "CUDA", "OpenCL", "C").
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Dev::Auto => "Auto",
            Dev::C => "C",
            Dev::Cblas => "CBLAS",
            Dev::Dummy => "Dummy",
            Dev::Cuda(_) => "CUDA",
            Dev::OpenCL(_) => "OpenCL",
            #[cfg(feature = "tenstorrent")]
            Dev::TT(_) => "Tenstorrent",
            Dev::Vulkan(_) => "Vulkan",
            #[cfg(feature = "wgpu")]
            Dev::WGPU(_) => "WGPU",
        }
    }

    /// Compile a kernel into a device program. Returns a program ID usable with
    /// `launch` and `release`. The `debug_asm` flag controls whether the backend
    /// prints the compiled assembly/source (for `ZYX_DEBUG=16`).
    pub fn compile(self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let result = match self {
            Dev::Auto => panic!("Dev::Auto cannot compile; resolve it with Dev::auto() first"),
            Dev::C => c::device().expect("C device unavailable").lock().unwrap().compile(kernel, debug_asm),
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().compile(kernel, debug_asm),
            Dev::Dummy => dummy::device().expect("dummy device unavailable").lock().unwrap().compile(kernel, debug_asm),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).compile(kernel, debug_asm),
            Dev::OpenCL(id) => dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).compile(kernel, debug_asm),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => dlock(self, &tenstorrent::device(id).expect("TT device unavailable")).compile(kernel, debug_asm),
            Dev::Vulkan(id) => dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).compile(kernel, debug_asm),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).compile(kernel, debug_asm),
        };
        if let Ok(x) = std::env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
            && DebugMask(x).compile()
        {
            println!("[{}] compile kernel", self.name());
        }
        result
    }

    /// Free a compiled program and its device resources (pipeline, shader module, etc.).
    pub fn release(self, program_id: DeviceProgramId) {
        match self {
            Dev::Auto => panic!("Dev::Auto cannot release; resolve it with Dev::auto() first"),
            Dev::C => c::device().expect("C device unavailable").lock().unwrap().release(program_id),
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().release(program_id),
            Dev::Dummy => dummy::device().expect("dummy device unavailable").lock().unwrap().release(program_id),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).release(program_id),
            Dev::OpenCL(id) => dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).release(program_id),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => dlock(self, &tenstorrent::device(id).expect("TT device unavailable")).release(program_id),
            Dev::Vulkan(id) => dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).release(program_id),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).release(program_id),
        }
    }

    /// Pattern-matches subgraphs in `graph` (e.g. matmul) and adds `Node::Kernel`s
    /// backed by this device's AOT kernels so they compete with the fused zyx
    /// kernels in extraction. No-op for devices without AOT kernels.
    pub fn match_graph(self, graph: &mut Graph, outputs: &BTreeSet<ClassId>) {
        match self {
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().match_graph(graph, outputs),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).match_graph(graph, outputs),
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
    /// parameter. `LaunchArg::Buffer` ids point into this device's pool
    /// ([`Dev::pool`]); `LaunchArg::Variable` carries the scalar
    /// value directly — backends never store variables. The grid (gws) is NOT
    /// passed here — each backend derives it at launch from the per-axis
    /// `GwsDim` it stored at compile, evaluating `Param(ordinal)` leaves
    /// against `args[ordinal]` (`LaunchArg::Variable` → `Constant::as_dim()`).
    pub fn launch(self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<(), BackendError> {
        // A kernel always has at least one Param (its output); launching with
        // no args means buffer binding failed upstream — backends would pass
        // garbage param pointers to the driver.
        debug_assert!(!args.is_empty(), "launch with empty args: buffer binding failed upstream");
        let pool = self.pool();
        match self {
            Dev::Auto => panic!("Dev::Auto cannot launch; resolve it with Dev::auto() first"),
            Dev::C => c::device().expect("C device unavailable").lock().unwrap().launch(program_id, pool, args),
            Dev::Cblas => cblas::device().expect("CBLAS device unavailable").lock().unwrap().launch(program_id, pool, args),
            Dev::Dummy => dummy::device().expect("dummy device unavailable").lock().unwrap().launch(program_id, pool, args),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).launch(program_id, pool, args),
            Dev::OpenCL(id) => {
                dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).launch(program_id, pool, args)
            }
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => dlock(self, &tenstorrent::device(id).expect("TT device unavailable")).launch(program_id, pool, args),
            Dev::Vulkan(id) => {
                dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).launch(program_id, pool, args)
            }
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).launch(program_id, pool, args),
        }
    }

    /// Timed launch for autotune: returns the kernel's run time in nanos.
    /// The measurement is uncontended — the device's pending work is
    /// submitted and drained before the timed kernel runs solo.
    pub fn launch_timed(self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        // A kernel always has at least one Param (its output); launching with
        // no args means buffer binding failed upstream — backends would pass
        // garbage param pointers to the driver.
        debug_assert!(!args.is_empty(), "launch_timed with empty args: buffer binding failed upstream");
        match self {
            Dev::Auto => panic!("Dev::Auto cannot launch; resolve it with Dev::auto() first"),
            Dev::Cuda(id) => dlock(self, &cuda::device(id).expect("CUDA device unavailable")).launch_timed(program_id, args),
            Dev::OpenCL(id) => {
                dlock(self, &opencl::device(id).expect("OpenCL device unavailable")).launch_timed(program_id, args)
            }
            Dev::C => {
                let pool = self.pool();
                c::device().expect("C device unavailable").lock().unwrap().launch_timed(program_id, pool, args)
            }
            Dev::Cblas => todo!("launch_timed not yet ported to the CBLAS device"),
            Dev::Dummy => todo!("launch_timed not yet ported to the dummy device"),
            Dev::Vulkan(id) => {
                dlock(self, &vulkan::device(id).expect("Vulkan device unavailable")).launch_timed(program_id, args)
            }
            #[cfg(feature = "tenstorrent")]
            Dev::TT(id) => todo!("launch_timed not yet ported to the TT device ({id})"),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(id) => dlock(self, &wgpu::device(id).expect("WGPU device unavailable")).launch_timed(program_id, args),
        }
    }
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
    pub fn allocate(self, bytes: Dim) -> Result<PoolBufferId, BackendError> {
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

    /// Insert an already-filled host buffer into the pool. Only valid for
    /// [`Pool::Host`]; any other pool is a programming error and panics.
    pub fn insert_host(self, buf: Box<[u8]>) -> PoolBufferId {
        match self {
            Pool::Host => lock(self, &host::pool()).insert(buf),
            _ => unreachable!("Pool::insert is only valid for the host pool, got {self:?}"),
        }
    }

    /// Map a slice of a file on disk into the disk pool. Only valid for
    /// [`Pool::Disk`]; any other pool is a programming error and panics.
    pub fn disk_buffer_from_path(self, bytes: Dim, path: &std::path::Path, offset_bytes: u64) -> PoolBufferId {
        match self {
            Pool::Disk => lock(self, &disk::pool()).buffer_from_path(bytes, path, offset_bytes),
            _ => unreachable!("Pool::disk_buffer_from_path is only valid for the disk pool, got {self:?}"),
        }
    }

    /// Increment the buffer's reference count (allocate starts it at 1).
    /// Checked math: overflow panics.
    pub fn retain(self, buffer_id: PoolBufferId) {
        match self {
            Pool::Cuda(id) => {
                if let Ok(pool) = cuda::pool(id) {
                    lock(self, &pool).retain(buffer_id);
                }
            }
            Pool::Host => lock(self, &host::pool()).retain(buffer_id),
            // Disk buffers are file mappings — nothing to free behind rc.
            Pool::Disk => {}
            Pool::OpenCL(id) => {
                if let Ok(pool) = opencl::pool(id) {
                    lock(self, &pool).retain(buffer_id);
                }
            }
            Pool::Vulkan(id) => {
                if let Ok(pool) = vulkan::pool(id) {
                    lock(self, &pool).retain(buffer_id);
                }
            }
            Pool::Dummy => {
                if let Ok(pool) = dummy::pool() {
                    lock(self, &pool).retain(buffer_id);
                }
            }
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => {
                if let Ok(pool) = tenstorrent::pool(id) {
                    lock(self, &pool).retain(buffer_id);
                }
            }
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                if let Ok(pool) = wgpu::pool(id) {
                    lock(self, &pool).retain(buffer_id);
                }
            }
        }
    }

    /// Decrement the buffer's reference count. At zero the backend defers the
    /// actual reclamation behind all in-flight work (release is a request;
    /// freeing is backend-scheduled).
    pub fn release(self, buffer_id: PoolBufferId) {
        match self {
            Pool::Cuda(id) => {
                if let Ok(pool) = cuda::pool(id) {
                    lock(self, &pool).release(buffer_id);
                }
            }
            Pool::Host => lock(self, &host::pool()).release(buffer_id),
            // Disk buffers are file mappings — nothing to free behind rc.
            Pool::Disk => {}
            Pool::OpenCL(id) => {
                if let Ok(pool) = opencl::pool(id) {
                    lock(self, &pool).release(buffer_id);
                }
            }
            Pool::Vulkan(id) => {
                if let Ok(pool) = vulkan::pool(id) {
                    lock(self, &pool).release(buffer_id);
                }
            }
            Pool::Dummy => {
                if let Ok(pool) = dummy::pool() {
                    lock(self, &pool).release(buffer_id);
                }
            }
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => {
                if let Ok(pool) = tenstorrent::pool(id) {
                    lock(self, &pool).release(buffer_id);
                }
            }
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                // Free must not race unsubmitted launches that still use the
                // buffer.
                if wgpu::flush_pending(id).is_ok()
                    && let Ok(pool) = wgpu::pool(id)
                {
                    lock(self, &pool).release(buffer_id);
                }
            }
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

    pub fn pool_to_host(self, src: PoolBufferId, dst: &mut [u8]) -> Result<(), BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).pool_to_host(src, dst),
            Pool::Disk => lock(self, &disk::pool()).pool_to_host(src, dst),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).pool_to_host(src, dst),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).pool_to_host(src, dst),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).pool_to_host(src, dst),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).pool_to_host(src, dst),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                wgpu::flush_pending(id)?;
                lock(self, &wgpu::pool(id)?).pool_to_host(src, dst)
            }
            Pool::Dummy => lock(self, &dummy::pool()?).pool_to_host(src, dst),
        }
    }

    /// Copy data from `src` pool into `self` (dst). The source buffer is
    /// retained for the duration of the async copy and released by the
    /// destination backend once the copy completes. Lock order is always
    /// dst-then-src; both are separate mutexes so this cannot deadlock as long
    /// as no path locks them in the opposite order.
    pub fn pool_to_pool(self, src: Pool, src_buf: PoolBufferId, dst_buf: PoolBufferId) -> Result<(), BackendError> {
        match self {
            Pool::Host => lock(self, &host::pool()).pool_to_pool(src, src_buf, dst_buf),
            Pool::Disk => todo!("copies into disk pool"),
            Pool::Cuda(id) => lock(self, &cuda::pool(id)?).pool_to_pool(src, src_buf, dst_buf),
            Pool::OpenCL(id) => lock(self, &opencl::pool(id)?).pool_to_pool(src, src_buf, dst_buf),
            Pool::Vulkan(id) => lock(self, &vulkan::pool(id)?).pool_to_pool(src, src_buf, dst_buf),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => lock(self, &tenstorrent::pool(id)?).pool_to_pool(src, src_buf, dst_buf),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(id) => {
                wgpu::flush_pending(id)?;
                lock(self, &wgpu::pool(id)?).pool_to_pool(src, src_buf, dst_buf)
            }
            Pool::Dummy => lock(self, &dummy::pool()?).pool_to_pool(src, src_buf, dst_buf),
        }
    }
}

/// Process-wide backend config, parsed once from `$XDG_CONFIG_HOME/zyx/config.json`
/// (else `~/.config/zyx/config.json`) on first access. Missing or unparsable
/// file means defaults.
pub(crate) fn config() -> &'static Config {
    static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
    CONFIG.get_or_init(load_config_file)
}

/// Read the backend config file (same search `Runtime::initialize_backends`
/// used to do). Missing or unparsable file means defaults.
fn load_config_file() -> Config {
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

/// Autotune config from the backend config file.
/// Falls back to `BeamSearch::new()` defaults when the file is missing or
/// unparsable — the same values `Runtime` used before lazy init existed.
pub(crate) fn autotune_config() -> crate::kernel::autotune::BeamSearch {
    config().autotune.clone()
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

/// Globally unique buffer identifier: the owning global pool plus the
/// buffer id within that pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Buffer {
    pub pool: Pool,
    pub buffer_id: PoolBufferId,
}

impl Buffer {
    pub const NULL: Self = Self { pool: Pool::Host, buffer_id: PoolBufferId(u32::MAX) };
}

impl From<Buffer> for usize {
    fn from(value: Buffer) -> Self {
        value.buffer_id.0 as usize
    }
}

/// Globally unique program identifier: the owning device plus the
/// program id within that device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramId {
    pub dev: Dev,
    pub program_id: DeviceProgramId,
}

impl ProgramId {
    pub const NULL: Self = Self { dev: Dev::Auto, program_id: DeviceProgramId(u32::MAX) };
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
    /// Whether the C backend was compiled with OpenMP support.
    /// Only meaningful for the C device; false everywhere else.
    pub has_openmp: bool,
}

impl DeviceInfo {
    /// Returns operation capabilities for a dtype (none() if dtype is unsupported)
    pub const fn supports_dtype(&self, dtype: DType) -> DTypeCapability {
        self.dtype_capability[dtype as usize]
    }
}
