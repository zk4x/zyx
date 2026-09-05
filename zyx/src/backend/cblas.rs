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

use super::{
    DTypeCapability, Device, DeviceId, DeviceInfo, DeviceProgramId, Event, LaunchArg, MemoryPool, PoolId, ProgramId,
    host::HostEvent,
};
use crate::{
    DType, Set,
    error::{BackendError, ErrorStatus},
    graph::{ClassId, Graph, Node, NodeData},
    kernel::Kernel,
    shape::Dim,
    slab::{Slab, SlabId},
};
use libloading::Library;
use nanoserde::DeJson;
use std::{collections::BTreeSet, sync::Arc};

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
    device_id: DeviceId,
    memory_pool_id: PoolId,
    /// Keeps the libopenblas library loaded so the [`CblasKernel`] fn pointers stay valid.
    /// Never read, but dropping it would unload the library.
    #[allow(dead_code)]
    lib: Library,
    kernels: Slab<CblasKernelId, CblasKernel>,
    programs: Slab<DeviceProgramId, CblasProgram>,
}

pub(super) fn initialize_device(
    config: &CblasConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        if debug_dev {
            println!("[cblas] configured out");
        }
        return Ok(());
    }
    if memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "cblas backend requires HostMemoryPool to be initialized first.".into(),
        });
    }

    // Load libopenblas and fail fast if cblas_sgemm is missing
    let lib = unsafe { Library::new(OPENBLAS_PATH) }?;
    let sgemm: SgemmFn = *unsafe { lib.get(b"cblas_sgemm") }?;

    let mut kernels = Slab::new();
    kernels.push(CblasKernel { sgemm });

    let device_id = devices.push(Device::Cblas(CblasDevice {
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
        }),
        device_id: DeviceId::NULL,
        // cblas reuses the host pool (like the C backend)
        memory_pool_id: PoolId::from(0),
        lib,
        kernels,
        programs: Slab::new(),
    }));
    if let Device::Cblas(dev) = &mut devices[device_id] {
        dev.device_id = device_id;
    }
    if debug_dev {
        println!("[cblas] initialized from {OPENBLAS_PATH}");
    }
    Ok(())
}

impl CblasDevice {
    pub const fn deinitialize(&mut self) {}

    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
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
    pub fn match_graph(&mut self, graph: &mut Graph, outputs: &BTreeSet<ClassId>) {
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
            let nid = graph.nodes.push(NodeData {
                node: Node::Kernel {
                    inputs: Box::new([mm.a, mm.b]),
                    outputs: Box::new([mm.out]),
                    program_id: ProgramId { device_id: self.device_id, program_id },
                    time: 1,
                },
                class_of: mm.out,
            });
            graph.classes[mm.out].nodes.push(nid);
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut super::host::HostMemoryPool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list; // sync not needed for sequential CPU

        let program = &self.programs[program_id];
        let kernel = &self.kernels[program.kernel];

        let m: i32 = i32::try_from(program.m)
            .map_err(|_| BackendError { status: ErrorStatus::IncorrectKernelArg, context: "m exceeds i32 range".into() })?;
        let n: i32 = i32::try_from(program.n)
            .map_err(|_| BackendError { status: ErrorStatus::IncorrectKernelArg, context: "n exceeds i32 range".into() })?;
        let k: i32 = i32::try_from(program.k)
            .map_err(|_| BackendError { status: ErrorStatus::IncorrectKernelArg, context: "k exceeds i32 range".into() })?;

        // args are [a, b, out] — loads first, then stores
        let LaunchArg::Buffer(b0) = args[0] else {
            unreachable!("cblas sgemm args are plain buffers")
        };
        let LaunchArg::Buffer(b1) = args[1] else {
            unreachable!("cblas sgemm args are plain buffers")
        };
        let LaunchArg::Buffer(b2) = args[2] else {
            unreachable!("cblas sgemm args are plain buffers")
        };
        let a = memory_pool.buffer_ptr_mut(b0) as *mut f32;
        let b = memory_pool.buffer_ptr_mut(b1) as *mut f32;
        let c = memory_pool.buffer_ptr_mut(b2) as *mut f32;

        unsafe {
            // Row-major, NoTrans x NoTrans: C(m, n) = A(m, k) @ B(k, n)
            (kernel.sgemm)(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
        }

        Ok(Event::Host(HostEvent))
    }
}
