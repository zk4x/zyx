// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
// Tenstorrent backend for zyx.
//
// # Attribution
//
// The `dev/`, `kern/` and `sim/` subtrees are derived from **metalium-rs**
// (https://github.com/Knight-Ops/metalium-rs) by **Carl Hurd (Knight-Ops)**,
// licensed under the Apache License, Version 2.0
// (https://www.apache.org/licenses/LICENSE-2.0). Only the parts zyx needs were
// carried over; each file keeps its upstream documentation, and the derived
// files stay under the Apache-2.0 terms they arrived with. `dev/` is
// `crates/tt-device` (minus its ethernet module), `kern/` is the subset of
// `crates/tt-kernels` zyx emits (`runtime`, `datapath`, `sfpu`, `l1`, `code`,
// `loops`, `kind`, and the matmul constants), and `sim/` is
// `crates/tt-ttsim` plus `crates/tt-ttsim-sys`. `tt-isa` stays an external
// dependency: its instruction tables are generated from Tenstorrent's ISA
// specification and auditing them is not work worth redoing.
//
// # Grid indexing (gidx)
//
// Two kernel-index dims map to the core's logical coordinate: gidx0 → row
// (y), gidx1 → col (x). Blackhole P100a: 10 rows × 13 cols (140 physical
// workers minus the dispatch column; harvested boards shrink x). Source:
// tt-metal `core_descriptors/blackhole_140_arch.yaml`.
//
// Device access goes through the native runtime (`self::ttsim`): the ported
// `dev/` layer over the ported `sim/` transport, held by one owner thread
// because the simulator is `!Send`. Every op that touches the device blocks on
// that thread. There is no C++ and no dlopen: a launch hands `kern::runtime::run`
// the three role streams the emitter staged, waits for every role's `DONE`, and
// returns. The simulator itself is a Tenstorrent binary blob, found through
// `TT_TTSIM_LIB` or `vendor/libttsim_bh.so`; only its wrapper lives here.
//
// DRAM is host memory in this transport: a `Pool::TT` chunk is a `Vec<u8>`
// owned by the pool, the boundary moves carry bytes in and out of it, and the
// device only ever sees L1 (`runtime::Kernel::stage` and `read_back`). That
// keeps the hard invariant intact — a kernel never round-trips through DRAM
// while its roles run.

use self::ttsim::{NativeRun, run as run_native};
use super::{ChunkId, Cmd, CmdQueue, Dev, DeviceInfo, DeviceProgramId, GwsDim, Kernel, LaunchArg, Placement, PlanDim, Pool, Shard};
use crate::{
    DType, Map, Set, ZyxError, bf16, f16,
    backend::DTypeCapability,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{GPUOp, Op, OpId, TTOp},
    shape::Dim,
    slab::Slab,
};
use nanoserde::{DeJson, SerBin};
use std::sync::{Arc, Mutex, OnceLock};
use tt_isa::isa::Instruction;

// The ported layers are carried over whole so their behaviour matches
// upstream bit for bit; only the pieces zyx calls are exercised, so the rest
// is dead until a later pass prunes it.
#[allow(dead_code, unused_imports, unused_variables)]
pub(crate) mod dev;
#[allow(dead_code, unused_imports, unused_variables)]
pub(crate) mod kern;
#[allow(dead_code, unused_imports, unused_variables)]
pub(crate) mod sim;
pub(crate) mod ttsim;

use kern::code::Loop;
use kern::runtime::SemaphoreInit;

// ── Global state ──────────────────────────────────────────────────────────────

// The single backend global: pools + devices, built once. One `OnceLock` runs
// `init_global` exactly once — concurrent threads block until it returns and
// reuse the tables.
struct TTGlobal {
    backend: TTBackend,
}
static TT: OnceLock<TTGlobal> = OnceLock::new();

/// Pools and their devices, built together by [`initialize_backend`].
struct TTBackend {
    pools: Vec<Mutex<TTMemoryPool>>,
    devices: Vec<Mutex<TTDevice>>,
}

fn init_global() -> TTGlobal {
    TTGlobal { backend: initialize_backend() }
}

/// Builds pools + devices together in one pass and publishes the combined
/// table. Runs once.
fn initialize_backend() -> TTBackend {
    let config = super::config();
    let debug_dev = super::debug_backends();
    let pools = ensure_pool_table(&config.tenstorrent, debug_dev).expect("tenstorrent: pool table init failed");
    let mut devices = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::TT(u16::try_from(idx).expect("So many Tenstorrent devices..."));
        let guard = super::lock(pool_id, pool);
        devices.push(Mutex::new(TTDevice {
            device_info: Arc::new(guard.dev_info.clone()),
            memory_pool: pool_id,
            programs: Slab::new(),
        }));
        drop(guard);
    }
    TTBackend { pools, devices }
}

// ---------------------------------------------------------------------------
// DRAM size lookup
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// DRAM size lookup
// ---------------------------------------------------------------------------

/// GDDR6 sizes for known Blackhole PCI subsystem IDs (total per board).
/// P100/P100A: 7 usable channels → 28 GB; P150: 8 → 32 GB; P300: 2×8 → 64 GB.
/// The driver does not expose GDDR6 capacity, so this table is the fallback.
/// Sources: docs.tenstorrent.com/aibs/blackhole, tt-umd `board_upi_map`,
/// tt-metal `blackhole_140_arch.yaml` (dram_bank_size ≈ 4 GB).
const DRAM_SIZE_TABLE: &[(u16, &str, u64)] = &[
    (0x0036, "p100", 28u64 * 1024 * 1024 * 1024),
    (0x0040, "p150a", 32u64 * 1024 * 1024 * 1024),
    (0x0041, "p150b", 32u64 * 1024 * 1024 * 1024),
    (0x0042, "p150c", 32u64 * 1024 * 1024 * 1024),
    (0x0043, "p100a", 28u64 * 1024 * 1024 * 1024),
    (0x0044, "p300b", 64u64 * 1024 * 1024 * 1024),
    (0x0045, "p300a", 64u64 * 1024 * 1024 * 1024),
    (0x0046, "p300c", 64u64 * 1024 * 1024 * 1024),
];

fn detect_dram_bytes() -> u64 {
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        let found = entries.flatten().find_map(|entry| {
            let vendor = std::fs::read_to_string(entry.path().join("vendor")).ok()?;
            if vendor.trim() != "0x1e52" {
                return None;
            }
            let subsys = std::fs::read_to_string(entry.path().join("subsystem_device")).ok()?;
            u16::from_str_radix(subsys.trim().trim_start_matches("0x"), 16).ok()
        });
        if let Some(id) = found {
            for &(sid, _name, size) in DRAM_SIZE_TABLE {
                if sid == id {
                    return size;
                }
            }
        }
    }
    64u64 * 1024 * 1024 * 1024
}
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct TTConfig {
    /// If set to None, then it will automatically use all Tenstorrent devices,
    /// otherwise it uses only selected devices
    device_ids: Option<Vec<i32>>,
}

// ---------------------------------------------------------------------------
// Native program: what codegen stages and the backend runs.
//
// This is the whole kernel as the hardware executes it: data-mover moves
// across the DRAM/L1 boundary, the three Tensix role streams, the
// semaphores they synchronize on, and the build-time L1/DST placement the
// instructions bake in as immediates. It lives here (not in the kernel IR)
// beside the device types it feeds; `TTProgramDesc` carries it in the IR so
// `compile` can reach it after lowering.
//
// Addresses: L1 is a zyx-owned bump allocation over the metalium data arena
// (`tt_isa::l1::DATA`), first touch, 64-byte aligned. DRAM offsets are not
// known at compile time (`Pool::TT` chunks get their ranges at launch), so a
// DM move carries the param ordinal and the launch fills in
// channel/port/offset.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One data-mover move of the native program: `bytes` between the DRAM
/// buffer of param `param` and L1 address `l1`.
pub struct TTDmMove {
    /// Global head-order param ordinal (the shim/launch arg index).
    pub param: u32,
    /// L1 byte address of the CB side of the move.
    pub l1: u32,
    /// Transfer size in bytes of the L1 side.
    pub bytes: u32,
    /// Writer-side narrowing: the L1 tile is FP32 and the param is 16-bit,
    /// so the host write-back converts (`TT_NARROW_*`). The packer cannot do
    /// it: F32 `Dst` -> BF16 L1 is `pack_fmt_conv_mode=0x105`, which the
    /// pinned ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
    pub narrow: u8,
}

/// `TTDmMove::narrow`: the L1 bytes are copied as they are.
pub const TT_NARROW_NONE: u8 = 0;
/// `TTDmMove::narrow`: FP32 L1 tile -> FP16 param, round to nearest even.
pub const TT_NARROW_F16: u8 = 1;
/// `TTDmMove::narrow`: FP32 L1 tile -> BF16 param, round to nearest even.
pub const TT_NARROW_BF16: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One host-side block repeat, mirroring `tt_kernels::code::Loop`: the
/// firmware pushes `[start, start + len)` `count` times.
pub struct TTLoop {
    /// First instruction word of the repeated block.
    pub start: u32,
    /// Word count of the repeated block.
    pub len: u32,
    /// Repeat count.
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One semaphore's initial state, mirroring
/// `tt_kernels::runtime::SemaphoreInit` (`(Semaphore, value, max)`), stored
/// as plain bytes so the program stays `SerBin`.
pub struct TTSemaphoreInit {
    /// Semaphore index, `0..8`.
    pub index: u8,
    /// Initial value.
    pub value: u8,
    /// Maximum value.
    pub max: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One Register storage's physical DST row.
pub struct TTDstRow {
    /// Storage op (bits of its `OpId`).
    pub storage: u32,
    /// Absolute DST row the storage occupies.
    pub row: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// The staged native program. Empty vectors mean "not emitted yet": the C++
/// section source is still the fallback for anything the native emitter does
/// not cover.
pub struct TTNativeProgram {
    /// Reader DM moves, in program order (DRAM param -> L1 CB).
    pub reader: Vec<TTDmMove>,
    /// Writer DM moves, in program order (L1 CB -> DRAM param).
    pub writer: Vec<TTDmMove>,
    /// Compute role streams in T0/T1/T2 order, raw `Instruction::word()`s.
    /// `Instruction` holds a `&'static InstructionDef`, so only the word is
    /// staged.
    pub compute: [Vec<u32>; 3],
    /// Host-side block loops, per role, in the same T0/T1/T2 order.
    pub compute_loops: [Vec<TTLoop>; 3],
    /// Semaphore initial states for the run.
    pub sems: Vec<TTSemaphoreInit>,
    /// L1 byte address per CB, in CB index order (the same order as
    /// `TTProgramDesc::cb_config`).
    pub l1: Vec<u32>,
    /// Physical DST row per Register storage, sorted by row.
    pub dst: Vec<TTDstRow>,
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// A Tenstorrent device's memory.
//
// `Pool::TT` memory is host memory: the native transport stages the boundary
// with host copies, so a TT tensor lives in the host pool and `Pool::TT`'s
// memory operations delegate there (`backend::mod`). What a TT pool carries
// is the device: its capabilities and its compiled programs.
// ---------------------------------------------------------------------------

fn backend() -> Result<(&'static Vec<Mutex<TTMemoryPool>>, &'static Vec<Mutex<TTDevice>>), BackendError> {
    let g = TT.get_or_init(init_global);
    Ok((&g.backend.pools, &g.backend.devices))
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<TTMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::TT({id}) is not available").into() }
}

#[derive(Debug)]
pub struct TTMemoryPool {
    pub(crate) dev_info: DeviceInfo,
}

fn ensure_pool_table(config: &TTConfig, debug_dev: bool) -> Result<Vec<Mutex<TTMemoryPool>>, BackendError> {
    let mut pools: Vec<Mutex<TTMemoryPool>> = Vec::new();
    // Configured out: empty tables, no device, no thread.
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(pools);
    }

    // One simulated chip, which is device 0. More than one would need a
    // transport each, and the owner thread opens exactly one simulator.
    let devices = match &config.device_ids {
        Some(ids) => ids.len(),
        None => 1,
    };
    if devices > 1 {
        todo!("multi-device Tenstorrent on the native transport: {devices} devices configured, one simulated chip");
    }
    if let Some(ids) = &config.device_ids
        && ids.iter().any(|id| *id != 0)
    {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("tenstorrent device ids {ids:?}: the simulator has one chip, id 0").into(),
        });
    }
    // One tile's worth of grid: the Tensix columns and rows of a Blackhole
    // (`tt_isa::noc::grid`), which is what the simulator models.
    use tt_isa::noc::grid::{TENSIX_COLUMNS, TENSIX_ROW_COUNT};
    let rows = TENSIX_ROW_COUNT as u32;
    let cols = TENSIX_COLUMNS.len() as u32;
    let dram_bytes = detect_dram_bytes();

    // F8E5M2 has no Blackhole DataFormat: not a capable dtype, codegen rejects it.
    let mut dtype_capability = [DTypeCapability::all(); DType::N_DTYPES];
    dtype_capability[DType::F8E5M2 as usize] = DTypeCapability::ZERO;
    for idx in 0..devices {
        if debug_dev {
            println!("[tenstorrent] device {idx} initialized");
            println!("[tenstorrent] device total memory: {} MB", dram_bytes / (1024 * 1024));
            println!("[tenstorrent] tensix grid {rows} rows x {cols} cols");
        }
        pools.push(Mutex::new(TTMemoryPool {
            dev_info: DeviceInfo {
                compute: 200_000_000_000_000, // ~200 TFLOPS BF16
                // Grid axes only (gidx0 row, gidx1 col); TT launches at most
                // 2 group axes, and the launch path rejects more.
                max_global_work_dims: vec![Dim::from(rows), Dim::from(cols)],
                max_local_threads: 1024,
                max_local_work_dims: vec![1, 1024, 1],
                preferred_vector_size: 32,
                local_mem_size: 0, // no thread-shared scratchpad; L1 is per-core private
                max_register_bytes: 128,
                tensor_cores: true,
                warp_size: 1, // Tensix has no SIMT warps
                cc: [0, 0],
                dtype_capability,
                has_native_exp2: false,
                supported_vec_lens: vec![32],
                vector_depth: 2,
                tenstorrent: true,
                tile: [32, 32],
                tile_sizes: vec![[32, 32]],
                wmma_layouts: vec![],
                num_circular_buffers: 32, // architectural CB0-CB31
                has_openmp: false,
            },
        }));
    }

    Ok(pools)
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<TTDevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::TT({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}

#[derive(Debug)]
struct TTProgram {
    /// The staged program, ready to hand to the owner thread: the three role
    /// streams decoded to `Instruction`s, their loops, the semaphore init and
    /// the boundary moves.
    native: TTNativeProgram,
    roles: [Vec<Instruction>; 3],
    loops: [Vec<Loop>; 3],
    sems: Vec<SemaphoreInit>,
    /// Global param count (inputs) and GlobalMut count (outputs).
    n_inputs: u32,
    n_outputs: u32,
    /// Group-range lengths in axis order: Const resolved at compile,
    /// Param(ordinal) resolved from the launch args.
    gws: Vec<GwsDim>,
    /// Tensix grid rows/cols at compile: dynamic launch sizes check here.
    max_grid: [u32; 2],
}

#[derive(Debug)]
pub struct TTDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
    programs: Slab<DeviceProgramId, TTProgram>,
}

/// Decode one native instruction word back to its definition, for the
/// runtime's `Instruction`. The staged program keeps words only (`SerBin`),
/// so the identity is recovered here: `InstructionDef::matches` is opcode
/// plus fixed fields, and the first match is the definition to push.
///
/// A definition's identity is not consulted by the runtime — it pushes
/// `word()` — so a shared opcode picking its first form is harmless; a word
/// the ISA does not know at all is a compile error, never a silent zero.
fn tt_decode(role: usize, word: u32) -> Result<Instruction, BackendError> {
    use tt_isa::isa::generated::ALL;
    let def = ALL.iter().copied().find(|def| def.matches(word)).ok_or_else(|| BackendError {
        status: ErrorStatus::KernelCompilation,
        context: format!("native T{role} word {word:#010x} decodes as no Tenstorrent instruction").into(),
    })?;
    Ok(Instruction::new(word, def))
}

impl TTDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        // Zero until TT codegen covers the full op/grid space: Dev::Auto
        // ranks by free_compute, and an auto-selected TT device fails
        // compilation (e.g. grid axis restrictions) where C/CUDA succeed.
        // Explicit Dev::TT is unaffected.
        0
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // Render to descriptor; decode head positions. CB ids, section
        // params, param ordinals, and io counts arrive in the descriptor
        // tables; what stays here is launch-side assembly: the decoded grid,
        // the runtime CB config split, and the program compile call.
        let rendered = kernel.render()?;
        let mut order = rendered.ops_in_order();
        let Some(Op::TT(TTOp::ProgramDesc(desc))) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "head op is not ProgramDesc".into() });
        };
        let Some(GPUOp::Params(_params)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "second op is not Params".into() });
        };
        let Some(Op::TT(TTOp::TensixGrid(grid))) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "third op is not TensixGrid".into() });
        };
        let Some(Op::Source(reader)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "fourth op is not Source".into() });
        };
        let Some(Op::TT(TTOp::EndReader)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "fifth op is not EndReader".into() });
        };
        let Some(Op::Source(compute)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "sixth op is not Source".into() });
        };
        let Some(Op::TT(TTOp::EndCompute)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "seventh op is not EndCompute".into() });
        };
        let Some(Op::Source(writer)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "eighth op is not Source".into() });
        };

        // Per-section params (0 = reader, 1 = compute, 2 = writer): param
        // ordinals in ascending head order — Global + Variable interleaved
        // first, GlobalMut at the tail — then the core's grid coordinates
        // gidx0 (row) / gidx1 (col). See `Kernel::generate_tenstorrent` and
        // `tt_runtime_shim.cpp` `section_rt_args` for the consumption side.
        let n_params = desc.n_params;
        // Tensix grid from the descriptor (rank <= 2 enforced at render):
        // Param-backed lengths resolve at launch from the Variable arg.
        let gws: Vec<GwsDim> = grid.to_vec();

        let reader = reader.as_str().as_bytes();
        let reader_params: &[u32] = &desc.reader_params;
        let compute = compute.as_str().as_bytes();
        let compute_params: &[u32] = &desc.compute_params;
        let writer = writer.as_str().as_bytes();
        let writer_params: &[u32] = &desc.writer_params;
        if debug_asm {
            eprintln!("[tenstorrent] reader:\n{}", String::from_utf8_lossy(reader));
            eprintln!("[tenstorrent] compute:\n{}", String::from_utf8_lossy(compute));
            eprintln!("[tenstorrent] writer:\n{}", String::from_utf8_lossy(writer));
        }

        let native = desc.native.clone();
        if debug_asm {
            eprintln!("[tenstorrent] reader:\n{}", String::from_utf8_lossy(reader));
            eprintln!("[tenstorrent] compute:\n{}", String::from_utf8_lossy(compute));
            eprintln!("[tenstorrent] writer:\n{}", String::from_utf8_lossy(writer));
            for (role, words) in native.compute.iter().enumerate() {
                eprintln!("[tenstorrent] native T{role}: {} words", words.len());
            }
            eprintln!("[tenstorrent] native dm: {} reader, {} writer", native.reader.len(), native.writer.len());
        }
        let _ = (reader_params, compute_params, writer_params, n_params);

        // Snapshot the grid for the launch-time bounds check (dynamic
        // sizes only; const sizes already failed at compile above).
        let mg = &self.device_info.max_global_work_dims;
        let max_grid = [
            u32::try_from(mg[0]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid rows do not fit u32".into(),
            })?,
            u32::try_from(mg[1]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid cols do not fit u32".into(),
            })?,
        ];

        // The native program is the program: decode each role's words once,
        // here, so a word the ISA cannot name is a compile error.
        let mut roles: [Vec<Instruction>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (role, words) in native.compute.iter().enumerate() {
            let mut decoded = Vec::with_capacity(words.len());
            for word in words {
                decoded.push(tt_decode(role, *word)?);
            }
            roles[role] = decoded;
        }
        let mut loops: [Vec<Loop>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (role, entries) in native.compute_loops.iter().enumerate() {
            loops[role] = entries
                .iter()
                .map(|l| Loop { start: l.start, len: l.len, count: l.count })
                .collect();
        }
        let mut sems: Vec<SemaphoreInit> = Vec::with_capacity(native.sems.len());
        for s in &native.sems {
            let sem = tt_isa::sync::Semaphore::new(s.index).ok_or_else(|| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("semaphore {} is outside 0..8", s.index).into(),
            })?;
            sems.push((sem, s.value, s.max));
        }
        Ok(self.programs.push(TTProgram {
            native,
            roles,
            loops,
            sems,
            n_inputs: desc.n_inputs,
            n_outputs: desc.n_outputs,
            gws,
            max_grid,
        }))
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_id(program_id) {
            // The owner thread holds nothing per program: everything a run
            // needs is pushed into the tile's mailbox each launch. Plain
            // `remove` drops the staged program in place — `remove_and_return`
            // would move it out (push + swap) only for us to drop it.
            self.programs.remove(program_id);
        }
    }

    pub fn launch(&mut self, program_id: DeviceProgramId, pool_handle: Pool, args: &[LaunchArg]) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let prog = if self.programs.contains_id(program_id) {
            &self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };

        let n_inputs = prog.n_inputs as usize;
        let n_outputs = prog.n_outputs as usize;

        // One arg per param, head order: Global + Variable interleaved,
        // GlobalMut at the tail. Kinds derive from the args: Variable ->
        // Variable; Buffer above the GlobalMut tail -> Global.
        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!(
                    "expected {} args ({} inputs + {} outputs), got {}",
                    n_inputs + n_outputs,
                    n_inputs,
                    n_outputs,
                    args.len()
                )
                .into(),
            });
        }
        let n_params = args.len();
        debug_assert!(n_params >= n_inputs + n_outputs, "tt launch: {n_params} args for {n_inputs} inputs + {n_outputs} outputs");

        let globalmut_start = n_params - n_outputs;
        // Buffer args by param ordinal; Variable params carry their value.
        let mut chunks: Vec<Option<ChunkId>> = vec![None; n_params];
        let mut vars: Vec<(u32, u32)> = Vec::new();
        for (ordinal, arg) in args.iter().enumerate() {
            let ordinal_u32 = ordinal as u32;
            match arg {
                LaunchArg::Buffer { chunk, .. } => chunks[ordinal] = Some(*chunk),
                LaunchArg::Variable(value) => {
                    debug_assert!(
                        ordinal < globalmut_start,
                        "tt launch: Variable arg at ordinal {ordinal} inside GlobalMut tail"
                    );
                    let dim = value.as_dim().expect("variable launch arg has a concrete dim");
                    let v = u32::try_from(dim).map_err(|_| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("param {ordinal} variable value {dim} does not fit u32").into(),
                    })?;
                    vars.push((ordinal_u32, v));
                }
            }
        }

        // Grid dims from the group-range lengths (gws), in axis order.
        // Variable ordinals index the launch args (head order), which always
        // carry a Variable value at a Variable param's position.
        let mut grid_dims = [1u32, 1u32];
        if prog.gws.len() > 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tenstorrent supports at most 2 group axes, got {}", prog.gws.len()).into(),
            });
        }
        for (axis, g) in prog.gws.iter().enumerate() {
            let dim = g.eval(&mut |ordinal| {
                vars.iter()
                    .find(|(o, _)| *o == ordinal as u32)
                    .map(|(_, v)| Dim::from(*v))
                    .expect("gws param ordinal has no Variable launch arg")
            });
            grid_dims[axis] = u32::try_from(dim).map_err(|_| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("gws axis {axis} dim {dim} does not fit u32").into(),
            })?;
            // Dynamic sizes skip the compile check: bound them here.
            if grid_dims[axis] > prog.max_grid[axis] {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!(
                        "tenstorrent grid axis {axis} size {} exceeds device grid {}",
                        grid_dims[axis], prog.max_grid[axis]
                    )
                    .into(),
                });
            }
        }
        if crate::debug_mask().launch() {
            println!("[tt] launch {program_id:?} grid=({},{}) args={args:?}", grid_dims[0], grid_dims[1]);
        }
        if grid_dims != [1, 1] {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!(
                    "tenstorrent grid dispatch is not wired yet: this launch wants {} x {} tiles and the runtime programs one",
                    grid_dims[0], grid_dims[1]
                )
                .into(),
            });
        }
        // The boundary: reader moves copy the param's DRAM bytes into the CB's
        // L1 address before any role starts; writer moves read them back after
        // every role finished. Both stay outside the kernel, so no role ever
        // touches DRAM.
        let mut stage: Vec<(u64, Vec<u8>)> = Vec::with_capacity(prog.native.reader.len());
        {
            let host = super::host::pool();
            let hpool = super::lock(Pool::Host, host);
            for mv in &prog.native.reader {
                let chunk = chunks[mv.param as usize].ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("reader move names param {}, which has no buffer arg", mv.param).into(),
                })?;
                let data = hpool.get_buffer(chunk);
                let bytes = mv.bytes as usize;
                if bytes > data.len() {
                    return Err(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!(
                            "reader move of {bytes} bytes from param {} exceeds its {} byte buffer",
                            mv.param,
                            data.len()
                        )
                        .into(),
                    });
                }
                stage.push((u64::from(mv.l1), data[..bytes].to_vec()));
            }
        }
        let spans: Vec<(u64, usize, u32)> = prog
            .native
            .writer
            .iter()
            .map(|mv| (u64::from(mv.l1), mv.bytes as usize, mv.param))
            .collect();        let read_back: Vec<(u64, usize)> = spans.iter().map(|(l1, bytes, _)| (*l1, *bytes)).collect();
        let run = NativeRun {
            roles: prog.roles.clone(),
            loops: prog.loops.clone(),
            stage,
            read_back,
            sems: prog.sems.clone(),
        };
        let outcome = run_native(run).map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: e.into_boxed_str(),
        })?;
        let host = super::host::pool();
        let mut hpool = super::lock(Pool::Host, host);
        for (idx, (_, bytes, param)) in spans.iter().enumerate() {
            let chunk = chunks[*param as usize].ok_or_else(|| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("writer move names param {param}, which has no buffer arg").into(),
            })?;
            let out = &outcome.l1[idx];
            let dst = hpool.buffer_ptr_mut(chunk);
            match prog.native.writer[idx].narrow {
                TT_NARROW_NONE => {
                    let n = (*bytes).min(out.len());
                    // SAFETY: `dst` points at the destination chunk's bytes,
                    // which the host pool owns, and `n` never exceeds the
                    // read-back range.
                    unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), dst, n) };
                }
                TT_NARROW_BF16 => {
                    // SAFETY: `out` is the read-back L1 tile, four bytes per
                    // datum, and the writer CB holds whole tiles.
                    let datums = unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<f32>(), out.len() / 4) };
                    let n = datums.len() * 2;
                    if n > hpool.get_buffer(chunk).len() {
                        return Err(BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: format!(
                                "writer move narrows {} datums to {n} bytes, more than param {param}'s buffer",
                                datums.len()
                            )
                            .into(),
                        });
                    }
                    // SAFETY: the destination chunk owns at least `n` bytes
                    // (checked above) and is not aliased here.
                    let out16 = unsafe { std::slice::from_raw_parts_mut(dst.cast::<bf16>(), datums.len()) };
                    for (o, &d) in out16.iter_mut().zip(datums) {
                        *o = bf16::from_f32(d);
                    }
                }
                TT_NARROW_F16 => {
                    // SAFETY: as for the BF16 arm.
                    let datums = unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<f32>(), out.len() / 4) };
                    let n = datums.len() * 2;
                    if n > hpool.get_buffer(chunk).len() {
                        return Err(BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: format!(
                                "writer move narrows {} datums to {n} bytes, more than param {param}'s buffer",
                                datums.len()
                            )
                            .into(),
                        });
                    }
                    // SAFETY: as for the BF16 arm.
                    let out16 = unsafe { std::slice::from_raw_parts_mut(dst.cast::<f16>(), datums.len()) };
                    for (o, &d) in out16.iter_mut().zip(datums) {
                        *o = f16::from_f32(d);
                    }
                }
                other => unreachable!("writer narrowing code {other} is not TT_NARROW_NONE/F16/BF16"),
            }
        }
        Ok(())
    }

    /// Timed launch for autotune. Returns the kernel's run time in nanos.
    /// Tenstorrent launches are synchronous (the shim blocks through
    /// Finish), so a wall-clock bracket is already an uncontended
    /// measurement.
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let start = std::time::Instant::now();
        self.launch(program_id, self.memory_pool, args)?;
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// Copy a transfer into this device's pool.
    ///
    /// Every chunk is host memory (`Pool::TT` memory operations delegate to
    /// the host pool, see the pool note above), so this is one host-to-host
    /// byte move whatever the source device is: read the source through its
    /// own pool, write the destination chunk with the host pool's pointer.
    /// Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        debug_assert!(bytes >= 0, "TT copy of negative bytes");
        let [src_shard] = &src.shards[..] else {
            todo!("TT copy of multi-shard source placement")
        };
        let [dst_shard] = &dst.shards[..] else {
            todo!("TT copy of multi-shard destination placement")
        };
        debug_assert_eq!(dst_shard.pool, self.memory_pool, "TT copy destination is not on this device");
        debug_assert_eq!(src_shard.offset, 0, "TT copy source has a nonzero offset");
        debug_assert_eq!(dst_shard.offset, 0, "TT copy destination has a nonzero offset");
        if src_shard.pool == self.memory_pool {
            // Same device: a placement copy is a no-op (the chunk is the same).
            return Ok(());
        }
        let mut buf = vec![0u8; bytes as usize];
        src_shard.pool.pool_to_host(src_shard.chunk, &mut buf)?;
        let host = super::host::pool();
        let mut hpool = super::lock(Pool::Host, host);
        let dst_ptr = hpool.buffer_ptr_mut(dst_shard.chunk);
        // SAFETY: `dst_ptr` points at the destination chunk's bytes, which the
        // host pool owns and which `bytes` sized the allocation for; `buf`
        // holds exactly `bytes` readable bytes.
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), dst_ptr, buf.len()) };
        Ok(())
    }
}

/// Preplanned Tenstorrent partition: ordered commands plus per-command death
/// lists (slots whose final read is that command and which nothing later
/// needs). Replay allocates unbound defs on the fly and launches programs
/// back-to-back — launches are synchronous, so program order is the order.
#[derive(Debug)]
pub(crate) struct TTPartition {
    pub(crate) cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    pub(crate) dev: u16,
}

impl TTDevice {
    /// Schedule a command run for a Tenstorrent device. A slot dies at its
    /// last use unless pinned (a plan output or read after this partition).
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>, dev: u16) -> TTPartition {
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
        TTPartition { cmds, deaths, dev }
    }
}

impl TTPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut TTDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        let my_pool = dev.memory_pool;
        debug_assert_eq!(my_pool, Pool::TT(self.dev), "TT partition replayed on the wrong device");
        for (idx, cmd) in self.cmds.iter().enumerate() {
            match cmd {
                Cmd::Launch { program, args, outputs } => {
                    debug_assert_eq!(program.dev, Dev::TT(self.dev), "TT partition holds a non-TT program");
                    for (slot, dtype, dims) in outputs {
                        if resolved.contains_key(slot) {
                            continue;
                        }
                        let el = Dim::from(dtype.bit_size() / 8);
                        let bytes = dims.iter().map(|d| d.eval(vars)).fold(el, |a, b| a * b);
                        debug_assert!(bytes >= 0, "TT replay allocated negative bytes");
                        let chunk = my_pool.allocate(bytes)?;
                        resolved.insert(
                            *slot,
                            Arc::new(Placement { shards: vec![Shard { pool: my_pool, chunk, offset: 0, len: bytes as usize }] }),
                        );
                    }
                    // Buffers name their chunk (offsets are 0 — no sub-chunk
                    // views exist yet); variables carry their scalar value.
                    let mut launch_args: Vec<LaunchArg> = Vec::with_capacity(args.len());
                    for arg in args {
                        if let Some(placement) = resolved.get(arg) {
                            let [shard] = &placement.shards[..] else {
                                todo!("multi-shard slot in TT launch")
                            };
                            debug_assert_eq!(shard.pool, my_pool, "TT launch arg is not on this device");
                            debug_assert_eq!(shard.offset, 0, "TT launch arg has a nonzero offset");
                            launch_args.push(LaunchArg::Buffer { chunk: shard.chunk, offset: 0, len: shard.len });
                        } else if let Some(constant) = vars.get(arg) {
                            launch_args.push(LaunchArg::Variable(constant.clone()));
                        } else {
                            panic!("TT replay: launch arg {arg:?} is neither placed nor bound");
                        }
                    }
                    // Dry run skips execution (buffers stay uninitialized).
                    if std::env::var("ZYX_DRY_RUN").is_err() {
                        dev.launch(program.program_id, my_pool, &launch_args)?;
                    }
                }
                Cmd::Alias { class, to } => {
                    let placed = resolved.get(to).unwrap_or_else(|| panic!("TT replay: alias target {to:?} is unplaced")).clone();
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

// ── Host-side tile layout ─────────────────────────────────────────────────────
// Row-major logical bytes <-> physical face-order tile bytes, executed on
// the host during transfers (off-device; on-device tilize is later work).
// A tilized buffer stores each 32x32 tile contiguously (1024 elements),
// the four 16x16 faces inside each tile in face order (face row major:
// face = face_row * 2 + face_col, 256 elements each), outer dims batched
// outside the tiles. Sub-tile ranks lift historically: 0D -> [1, 1],
// 1D [N] -> row [1, N]; halo zero-fills (K-halo correctness for matmul).
// The two `Dev` layout transfers below are the only functions here:
// everything (staging copies, swizzles, physical sizing) is inline.

impl Dev {
    /// Copy a logical tensor into this device, honoring its layout. TT
    /// arm only, all other devices panic: layout transfers are a TT
    /// operation. Row-major sources stage through host temps, swizzle
    /// to face order on the host (zero-filled halo), and move as
    /// physical bytes; TT sources are already tile layout and copy
    /// physical bytes unchanged (tilize is not idempotent, tile layout
    /// never reorders). `shape` is the logical shape.
    pub(crate) fn tilize_transfer(
        &self,
        src: &Arc<Placement>,
        shape: &[Dim],
        dtype: DType,
    ) -> Result<Arc<Placement>, ZyxError> {
        let Dev::TT(_) = self else {
            panic!("tilize_transfer: not a TT device: {self:?}");
        };
        let el = dtype.bit_size() as usize / 8;
        debug_assert!(el > 0, "tilize_transfer: zero element width");
        let (rows, cols) = match shape {
            [.., r, c] => (*r, *c),
            [c] => (1, *c),
            [] => (1, 1),
        };
        debug_assert!(rows > 0 && cols > 0, "tilize_transfer: non-positive dims {rows}x{cols}");
        let (pr, pc) =
            self.info().expect("tilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(pr);
        phys_dims.push(pc);
        let [shard] = &src.shards[..] else {
            todo!("tilize_transfer: multi-shard source");
        };
        debug_assert_eq!(shard.offset, 0, "tilize_transfer: nonzero-offset source shard");
        if matches!(shard.pool, Pool::TT(_)) {
            // Already tile layout: physical byte copy, no swizzle.
            let mut queue = CmdQueue::new();
            queue.push(Cmd::Copy {
                src: OpId::from(0),
                dst: OpId::from(1),
                dst_pool: self.pool(),
                dst_dtype: dtype,
                dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
            });
            let mut boundary = Map::default();
            boundary.insert(OpId::from(0), Arc::clone(src));
            let mut outputs = Set::default();
            outputs.insert(OpId::from(1));
            let plan = queue.schedule(&outputs);
            let out = plan.replay(boundary, &Map::default())?;
            return Ok(Arc::clone(&out[&OpId::from(1)]));
        }
        // Logical bytes to a host temp through the existing copy path.
        let logical_len = (outer * rows * cols * el as Dim) as usize;
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: Pool::Host,
            dst_dtype: dtype,
            dst_dims: shape.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), Arc::clone(src));
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        let staged = Arc::clone(&out[&OpId::from(1)]);
        let [staged_shard] = &staged.shards[..] else {
            unreachable!("tilize_transfer: staged placement changed shape");
        };
        let mut logical = vec![0u8; logical_len];
        Pool::Host.pool_to_host(staged_shard.chunk, &mut logical)?;
        // Swizzle to face order on the host (zero-filled halo).
        let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
        let mut physical = vec![0u8; (outer * pr * pc * el as Dim) as usize];
        for b in 0..outer {
            for r in 0..rows {
                for c in 0..cols {
                    let (tr, tc) = (r / 32, c / 32);
                    let face = (r % 32) / 16 * 2 + (c % 32) / 16;
                    let within = (r % 16) * 16 + (c % 16);
                    let srci = ((b * rows + r) * cols + c) * el as Dim;
                    let dsti = (((b * ntr + tr) * ntc + tc) * 1024 + face * 256 + within) * el as Dim;
                    physical[dsti as usize..dsti as usize + el]
                        .copy_from_slice(&logical[srci as usize..srci as usize + el]);
                }
            }
        }
        // Physical bytes to this device.
        let plen = physical.len();
        let chunk = Pool::Host.insert_host(physical.into_boxed_slice());
        let staged = Arc::new(Placement {
            shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: plen }],
        });
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: self.pool(),
            dst_dtype: dtype,
            dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), staged);
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        Ok(Arc::clone(&out[&OpId::from(1)]))
    }

    /// Copy a TT buffer out of this device, honoring its layout. TT
    /// arm only, all other devices panic: layout transfers are a TT
    /// operation. TT destinations copy physical bytes unchanged; any
    /// other destination downloads physical bytes, unswizzles to
    /// logical row-major on the host, and moves logical bytes. `shape`
    /// is the logical shape.
    pub(crate) fn untilize_transfer(
        &self,
        src: &Arc<Placement>,
        shape: &[Dim],
        dtype: DType,
        dst_pool: Pool,
    ) -> Result<Arc<Placement>, ZyxError> {
        let Dev::TT(_) = self else {
            panic!("untilize_transfer: not a TT device: {self:?}");
        };
        let el = dtype.bit_size() as usize / 8;
        debug_assert!(el > 0, "untilize_transfer: zero element width");
        let (rows, cols) = match shape {
            [.., r, c] => (*r, *c),
            [c] => (1, *c),
            [] => (1, 1),
        };
        debug_assert!(rows > 0 && cols > 0, "untilize_transfer: non-positive dims {rows}x{cols}");
        let (pr, pc) =
            self.info().expect("untilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(pr);
        phys_dims.push(pc);
        let [shard] = &src.shards[..] else {
            todo!("untilize_transfer: multi-shard source");
        };
        debug_assert_eq!(shard.offset, 0, "untilize_transfer: nonzero-offset source shard");
        if matches!(dst_pool, Pool::TT(_)) {
            // Physical-to-physical: no layout change.
            let mut queue = CmdQueue::new();
            queue.push(Cmd::Copy {
                src: OpId::from(0),
                dst: OpId::from(1),
                dst_pool,
                dst_dtype: dtype,
                dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
            });
            let mut boundary = Map::default();
            boundary.insert(OpId::from(0), Arc::clone(src));
            let mut outputs = Set::default();
            outputs.insert(OpId::from(1));
            let plan = queue.schedule(&outputs);
            let out = plan.replay(boundary, &Map::default())?;
            return Ok(Arc::clone(&out[&OpId::from(1)]));
        }
        // Physical bytes to a host temp (the replay TT-source override
        // counts shard bytes; dims size the staging buffer).
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: Pool::Host,
            dst_dtype: dtype,
            dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), Arc::clone(src));
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        let staged = Arc::clone(&out[&OpId::from(1)]);
        let [staged_shard] = &staged.shards[..] else {
            unreachable!("untilize_transfer: staged placement changed shape");
        };
        let physical_len = (outer * pr * pc * el as Dim) as usize;
        let mut physical = vec![0u8; physical_len];
        Pool::Host.pool_to_host(staged_shard.chunk, &mut physical)?;
        // Unswizzle on the host, then move logical bytes.
        let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
        let mut logical = vec![0u8; (outer * rows * cols * el as Dim) as usize];
        for b in 0..outer {
            for r in 0..rows {
                for c in 0..cols {
                    let (tr, tc) = (r / 32, c / 32);
                    let face = (r % 32) / 16 * 2 + (c % 32) / 16;
                    let within = (r % 16) * 16 + (c % 16);
                    let srci = (((b * ntr + tr) * ntc + tc) * 1024 + face * 256 + within) * el as Dim;
                    let dsti = ((b * rows + r) * cols + c) * el as Dim;
                    logical[dsti as usize..dsti as usize + el]
                        .copy_from_slice(&physical[srci as usize..srci as usize + el]);
                }
            }
        }
        let logical_len = logical.len();
        let chunk = Pool::Host.insert_host(logical.into_boxed_slice());
        let staged = Arc::new(Placement {
            shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: logical_len }],
        });
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool,
            dst_dtype: dtype,
            dst_dims: shape.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), staged);
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        Ok(Arc::clone(&out[&OpId::from(1)]))
    }
}
