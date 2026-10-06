// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! CBLAS backend — pattern-matches matmul subgraphs and dispatches them to
//! `cblas_sgemm` from `libopenblas.so`.
//!
//! This backend does not compile generic zyx kernels. It only participates in
//! graph extraction via [`CblasDevice::match_graph`]: it finds canonical matmul
//! subgraphs (`out = a @ b`, `a: [m, k]`, `b: [k, n]`) and adds `Node::Kernel`s
//! with `time = 1`, which beat any fused zyx kernel in extraction. Only f32
//! (`cblas_sgemm`) is supported for now.
//!
//! The cblas device reuses the HostMemoryPool (like the C backend) and needs no
//! worker thread — `cblas_sgemm` is a blocking CPU call.

#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::needless_pass_by_ref_mut)]

use super::{Cmd, DTypeCapability, Dev, DeviceInfo, DeviceProgramId, Placement, Pool, ProgramId};
use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::Graph,
    kernel::{Kernel, Op, OpId},
    shape::Dim,
    slab::{Slab, SlabId},
};
use libloading::Library;
use nanoserde::DeJson;
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex, OnceLock},
};

// ── Global state ──────────────────────────────────────────────────────────────

static CBLAS_DEVICE: OnceLock<Mutex<CblasDevice>> = OnceLock::new();

/// `cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)`
type SgemmFn = unsafe extern "C" fn(
    order: i32,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
);

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

/// Hardcoded path to the OpenBLAS shared library for now.
const OPENBLAS_PATH: &str = "/usr/lib/x86_64-linux-gnu/libopenblas.so";

#[derive(Debug, DeJson)]
#[nserde(default)]
pub struct CblasConfig {
    /// Enable this backend
    pub enabled: bool,
}

impl Default for CblasConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CblasKernelId(u32);

impl From<usize> for CblasKernelId {
    fn from(value: usize) -> Self {
        CblasKernelId(u32::try_from(value).unwrap())
    }
}

impl From<CblasKernelId> for usize {
    fn from(value: CblasKernelId) -> Self {
        value.0 as usize
    }
}

impl SlabId for CblasKernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// An AOT cblas kernel. Currently just the gemm (`cblas_sgemm`) loaded from libopenblas.
/// The library is kept alive by [`CblasDevice::lib`].
#[derive(Debug)]
pub struct CblasKernel {
    sgemm: SgemmFn,
}

/// A dispatched gemm program. M, N and K are fixed at graph-match time.
#[derive(Debug)]
pub struct CblasProgram {
    kernel: CblasKernelId,
    m: Dim,
    n: Dim,
    k: Dim,
}

#[derive(Debug)]
pub struct CblasDevice {
    device_info: Arc<DeviceInfo>,
    /// Keeps the libopenblas library loaded so the [`CblasKernel`] fn pointers stay valid.
    /// Never read, but dropping it would unload the library.
    #[allow(dead_code)]
    lib: Library,
    kernels: Slab<CblasKernelId, CblasKernel>,
    programs: Slab<DeviceProgramId, CblasProgram>,
}

fn device_with(config: &CblasConfig, debug_dev: bool) -> Result<&'static Mutex<CblasDevice>, BackendError> {
    if let Some(dev) = CBLAS_DEVICE.get() {
        return Ok(dev);
    }
    if !config.enabled {
        if debug_dev {
            println!("[cblas] configured out");
        }
        return Err(BackendError { status: ErrorStatus::Initialization, context: "CBLAS backend configured out".into() });
    }
    // cblas reuses the global host pool — no init ordering needed.

    // Load libopenblas and fail fast if cblas_sgemm is missing
    let lib = unsafe { Library::new(OPENBLAS_PATH) }?;
    let sgemm: SgemmFn = *unsafe { lib.get(b"cblas_sgemm") }?;

    let mut kernels = Slab::new();
    kernels.push(CblasKernel { sgemm });

    let dev = Mutex::new(CblasDevice {
        // Tiny compute and no dtype capabilities: this device never gets picked
        // for generic (eager) kernels, it only runs matched AOT matmuls.
        device_info: Arc::new(DeviceInfo {
            compute: 1,
            max_global_work_dims: vec![Dim::from(0i64); 3],
            max_local_threads: 1,
            max_local_work_dims: vec![1, 1, 1],
            preferred_vector_size: 8,
            local_mem_size: 0,
            max_register_bytes: 0,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::none(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![],
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }),
        // cblas reuses the host pool (like the C backend)
        lib,
        kernels,
        programs: Slab::new(),
    });
    if debug_dev {
        println!("[cblas] initialized from {OPENBLAS_PATH}");
    }
    let _ = CBLAS_DEVICE.set(dev);
    Ok(CBLAS_DEVICE.get().unwrap())
}

pub(super) fn device() -> Result<&'static Mutex<CblasDevice>, BackendError> {
    device_with(&super::config().cblas, super::debug_backends())
}

/// Resolves a launch arg placement to a host pointer: the shard addressed to
/// the host pool.
fn host_ptr(memory_pool: &mut super::host::HostMemoryPool, placement: &Placement) -> *mut u8 {
    placement
        .shards
        .iter()
        .find(|shard| shard.pool == Pool::Host)
        .map(|shard| memory_pool.buffer_ptr_mut(shard.chunk))
        .expect("cblas launch arg has no host shard")
}

impl CblasDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    pub fn compile(&mut self, _kernel: &Kernel, _debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: "cblas device only runs AOT matmul kernels, it does not compile generic kernels.".into(),
        })
    }

    /// Pattern-matches matmul subgraphs in `graph` and adds `Node::Kernel`s backed
    /// by this device's gemm kernels so they compete with the fused zyx kernels in
    /// extraction. `time = 1` makes the AOT gemm win over any fused kernel.
    pub fn match_graph(&mut self, graph: &mut Graph, outputs: &BTreeSet<OpId>) {
        let order = graph.topo_sort_classes::<true>(&Set::default(), outputs, None);
        for &cid in &order {
            let Some(mm) = graph.match_matmul(cid) else {
                continue;
            };
            // Only f32 sgemm is loaded, so skip matmuls of any other dtype. The
            // a/b buffers are read as f32 and out is written as f32, so both the
            // operand and accumulate dtypes must be f32.
            if mm.in_dtype != DType::F32 || mm.acc_dtype != DType::F32 {
                continue;
            }
            println!("[cblas] matched matmul m={}, n={}, k={}", mm.m, mm.n, mm.k);
            let program_id = self.programs.push(CblasProgram { kernel: CblasKernelId::ZERO, m: mm.m, n: mm.n, k: mm.k });
            let inputs = graph.push_op(Op::Stack { ops: Box::new([mm.a, mm.b]) });
            let outputs = graph.push_op(Op::Stack { ops: Box::new([mm.out]) });
            graph.mint_node(
                Op::Program { inputs, outputs, info: Box::new((ProgramId { dev: Dev::Cblas, program_id }, 1)) },
                mm.out,
            );
        }
    }
}

/// Preplanned CBLAS partition: the ordered commands plus per-command
/// death lists. Mirrors [`super::c::CPartition`] — replay allocates
/// unbound defs on the fly from their specs, invokes sgemm
/// back-to-back, and drops dead slots per the lists.
#[derive(Debug)]
pub(crate) struct CblasPartition {
    cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
}

impl CblasDevice {
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>) -> CblasPartition {
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
        CblasPartition { cmds, deaths }
    }
}

impl CblasPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut CblasDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        // Sequential CPU: the kernel runs to completion before returning.
        let host = super::host::pool();
        let mut memory_pool = super::lock(Pool::Host, host);
        for (idx, cmd) in self.cmds.iter().enumerate() {
            match cmd {
                Cmd::Launch { program, args, outputs } => {
                    debug_assert_eq!(program.dev, Dev::Cblas, "CBLAS partition holds a non-CBLAS program");
                    for (slot, dtype, dims) in outputs {
                        if resolved.contains_key(slot) {
                            continue;
                        }
                        let bytes = dims.iter().map(|d| d.eval(vars)).fold(dtype.bit_size() as i64 / 8, |a, b| a * b);
                        debug_assert!(bytes >= 0, "CBLAS replay allocated negative bytes");
                        let chunk = memory_pool.allocate(bytes)?;
                        resolved.insert(
                            *slot,
                            Arc::new(Placement {
                                shards: vec![super::Shard { pool: Pool::Host, chunk, offset: 0, len: bytes as usize }],
                            }),
                        );
                    }
                    let program_ref = &dev.programs[program.program_id];
                    let kernel = &dev.kernels[program_ref.kernel];

                    let m: i32 = i32::try_from(program_ref.m).map_err(|_| BackendError {
                        status: ErrorStatus::IncorrectKernelArg,
                        context: "m exceeds i32 range".into(),
                    })?;
                    let n: i32 = i32::try_from(program_ref.n).map_err(|_| BackendError {
                        status: ErrorStatus::IncorrectKernelArg,
                        context: "n exceeds i32 range".into(),
                    })?;
                    let k: i32 = i32::try_from(program_ref.k).map_err(|_| BackendError {
                        status: ErrorStatus::IncorrectKernelArg,
                        context: "k exceeds i32 range".into(),
                    })?;

                    // args are [a, b, out] — loads first, then stores;
                    // sgemm takes plain buffers, never scalars.
                    let a =
                        host_ptr(&mut memory_pool, resolved.get(&args[0]).expect("cblas sgemm arg 0 is unplaced")) as *const f32;
                    let b =
                        host_ptr(&mut memory_pool, resolved.get(&args[1]).expect("cblas sgemm arg 1 is unplaced")) as *const f32;
                    let c =
                        host_ptr(&mut memory_pool, resolved.get(&args[2]).expect("cblas sgemm arg 2 is unplaced")) as *mut f32;

                    // Dry run: skip device execution, keep arg binding validation.
                    if std::env::var("ZYX_DRY_RUN").is_err() {
                        unsafe {
                            // Row-major, NoTrans x NoTrans: C(m, n) = A(m, k) @ B(k, n)
                            (kernel.sgemm)(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
                        }
                    }
                }
                Cmd::Alias { class, to } => {
                    let placed =
                        resolved.get(to).unwrap_or_else(|| panic!("CBLAS replay: alias target {to:?} is unplaced")).clone();
                    resolved.insert(*class, placed);
                }
                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
            }
            for dead in &self.deaths[idx] {
                resolved.remove(dead);
            }
        }
        Ok(())
    }
}
