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
// that thread. There is no C++ and no dlopen: compilation uploads a kernel's
// images into device DRAM once (`images`), a launch resolves them to their
// slots and the owner thread loads only what its residency ledger says is
// missing, then `kern::runtime::run` waits for every role's `DONE`. The
// simulator itself is a Tenstorrent binary blob, found through `TT_TTSIM_LIB`
// or `vendor/libttsim_bh.so`; only its wrapper lives here.
//
// A `Pool::TT` chunk is device DRAM (the card's GDDR6), not host memory: the
// pool hands out flat byte ranges of the chip's DRAM, and the host touches them
// only for placement and readback. Inside a launch the tile's data movers own
// the boundary: RISCV B reads the operands DRAM->L1 before the roles run and
// RISCV NC writes the results L1->DRAM after they finish (`tt_isa::dm`), so no
// tensor byte crosses the host. The device side only ever sees L1
// (`runtime::Kernel::stage` and `read_back`): a kernel never round-trips
// through DRAM while its roles run.

use self::ttsim::{NativeRun, run as run_native};
use super::{
    ChunkId, Cmd, CmdQueue, Dev, DeviceInfo, DeviceProgramId, GwsDim, Kernel, LaunchArg, Placement, PlanDim, Pool, Shard,
};
use crate::{
    DType, Map, Set, ZyxError,
    backend::DTypeCapability,
    bf16,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    f16,
    kernel::{GPUOp, Op, OpId, ParamKind, TTOp},
    shape::Dim,
    slab::{Slab, SlabId},
};
use nanoserde::{DeJson, SerBin};
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};
use tt_isa::dram::{Dram, DramRange};

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

use kern::runtime::SemaphoreInit;
use tt_isa::mailbox::role::Mailbox;
use tt_isa::tensix::Core;

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
            images: Slab::new(),
            firmware: None,
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
    pub device_ids: Option<Vec<i32>>,
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
    /// Byte offset of this move's slice inside the param's buffer at grid
    /// point `(0, 0)`.
    pub base: u32,
    /// Byte stride of the slice per grid index on axis 0 (gidx0, the row).
    pub gstride0: u32,
    /// Byte stride of the slice per grid index on axis 1 (gidx1, the column).
    pub gstride1: u32,
}

/// Byte offset of a move's slice for one grid point: the program is
/// tile-uniform, so this is the only thing a grid point changes.
pub fn tt_move_offset(mv: &TTDmMove, gidx0: u32, gidx1: u32) -> usize {
    mv.base as usize + gidx0 as usize * mv.gstride0 as usize + gidx1 as usize * mv.gstride1 as usize
}

/// `TTDmMove::narrow`: the L1 bytes are copied as they are.
pub const TT_NARROW_NONE: u8 = 0;
/// `TTDmMove::narrow`: FP32 L1 tile -> FP16 param, round to nearest even.
pub const TT_NARROW_F16: u8 = 1;
/// `TTDmMove::narrow`: FP32 L1 tile -> BF16 param, round to nearest even.
pub const TT_NARROW_BF16: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One semaphore's initial state, mirroring
/// `tt_kernels::runtime::SemaphoreInit` (`(Semaphore, value, max)`), stored
/// as plain bytes so the program table stays plain data.
pub struct TTSemaphoreInit {
    /// Semaphore index, `0..8`.
    pub index: u8,
    /// Initial value.
    pub value: u8,
    /// Maximum value.
    pub max: u8,
}

// ---------------------------------------------------------------------------
// Device DRAM
//
// A `Pool::TT` chunk is the card's DRAM: one flat byte range in the chip's
// GDDR6, handed out by this pool. The flat space is striped round-robin across
// the chip's usable channels in [`STRIPE`]-byte pages — Tenstorrent's own
// striping is simple round-robin, and that is sufficient. The allocator never
// has to know the channel geometry: every access translates the flat range it
// touches into per-channel [`DramRange`]s at I/O time, so chunks are arbitrary
// size and placement is one bump.
// ---------------------------------------------------------------------------

/// DRAM interleaving granularity: bytes striped round-robin across the chip's
/// channels before the next channel gets the next page. Tenstorrent interleaves
/// a buffer in units of its *page* — the tile image — so the page is
/// [`tt_isa::dm::TILE_SLOT`] (16-byte header + 1024 datums + pad), not the bare
/// 4 KB datum block: a tile image must land inside one channel range, because
/// `READ_TRANSPOSED`/`READ_BROADCAST_COL` are a single descriptor of one slot.
const STRIPE: u64 = tt_isa::dm::TILE_SLOT;

/// One allocation in the flat DRAM space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DramChunk {
    /// Flat byte address of the chunk's first byte.
    addr: u64,
    /// Chunk length in bytes, exactly as requested.
    len: u64,
}

fn io_error(status: ErrorStatus, msg: String) -> BackendError {
    BackendError { status, context: msg.into_boxed_str() }
}

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
    /// The chip's usable DRAM channels: the only way a [`DramRange`] is made.
    dram: Dram,
    /// Flat DRAM bytes this pool hands out.
    total: u64,
    free_bytes: Dim,
    chunks: Slab<ChunkId, DramChunk>,
    free_set: Set<ChunkId>,
    /// First flat address never handed out, [`STRIPE`] aligned.
    bump: u64,
}

impl TTMemoryPool {
    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    /// The chunk's length in bytes.
    pub fn chunk_len(&self, id: ChunkId) -> usize {
        usize::try_from(self.chunks[id].len).expect("DRAM chunk longer than a host slice")
    }

    /// Allocate `bytes` of device DRAM. Best-fit from the free list first
    /// (stable addresses across repeats), else fresh from the flat space —
    /// the host pool's policy, over a different address space.
    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.chunks[*id].len as Dim;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        let len: u64 =
            bytes.try_into().map_err(|_| io_error(ErrorStatus::MemoryAllocation, "DRAM allocation size too large".into()))?;
        let addr = self.bump;
        let end = addr.checked_add(len).ok_or_else(|| {
            io_error(ErrorStatus::MemoryAllocation, format!("DRAM allocation of {len} bytes overflows the flat space"))
        })?;
        if end > self.total {
            return Err(io_error(
                ErrorStatus::MemoryAllocation,
                format!("OOM: {len} bytes requested, {} bytes of {} free", self.free_bytes, self.total),
            ));
        }
        self.bump = end.next_multiple_of(STRIPE);
        self.free_bytes -= len as Dim;
        // Fresh chunks are not zeroed. The flat space is never handed out
        // twice through the bump, so a new chunk holds only what the card's
        // DRAM already had there (zeros on ttsim); a chunk reused from the
        // free list keeps its bytes, exactly like the host pool.
        Ok(self.chunks.push(DramChunk { addr, len }))
    }

    /// Put one chunk on the free list (reclaimable only by [`Self::dispose`]).
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.chunks.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown DRAM chunk {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of DRAM chunk {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Claim existing allocations, all or nothing.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Free every chunk on the free list at once; the ids become invalid.
    pub fn dispose(&mut self) {
        for id in core::mem::take(&mut self.free_set) {
            if let Some(chunk) = self.chunks.get(id) {
                self.free_bytes += chunk.len as Dim;
            }
            self.chunks.remove(id);
        }
    }

    /// Blocking upload: host bytes into the chunk's first bytes.
    pub fn host_to_pool(&mut self, src: &[u8], dst: ChunkId) -> Result<(), BackendError> {
        self.write_at(dst, 0, src)
    }

    /// Blocking download: the chunk's first bytes into host bytes (up to
    /// `dst.len()`).
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let len = self.chunk_len(src).min(dst.len());
        self.read_at(src, 0, &mut dst[..len])
    }

    /// Write `src` at byte `off` of chunk `id`.
    ///
    /// The bytes are gathered per channel and each channel's run goes out as
    /// one range, so a tensor of any size costs at most one DRAM write and one
    /// visibility barrier per channel (eight on a full chip), not one per page.
    pub fn write_at(&mut self, id: ChunkId, off: usize, src: &[u8]) -> Result<(), BackendError> {
        const STATUS: ErrorStatus = ErrorStatus::MemoryCopyH2P;
        let addr = self.chunks[id].addr;
        let off = u64::try_from(off).map_err(|_| io_error(STATUS, "DRAM write offset does not fit u64".into()))?;
        let len = src.len() as u64;
        self.check(id, off, len).map_err(|e| io_error(STATUS, e))?;
        let stripes = self.stripes(addr + off, len).map_err(|e| io_error(STATUS, e))?;
        let channel_count = usize::from(self.dram.channel_count());
        let mut runs: Vec<(DramRange, Vec<u8>)> = Vec::with_capacity(channel_count);
        let mut slot: Vec<Option<usize>> = vec![None; channel_count];
        for (range, at) in stripes {
            let channel = usize::from(range.channel().index());
            let index = match slot[channel] {
                Some(index) => index,
                None => {
                    slot[channel] = Some(runs.len());
                    runs.push((range, Vec::with_capacity(range.len() as usize)));
                    runs.len() - 1
                }
            };
            let (run, bytes) = &mut runs[index];
            let start = run.offset();
            debug_assert_eq!(start + bytes.len() as u64, range.offset(), "a channel's pages are contiguous");
            bytes.extend_from_slice(&src[at..at + range.len() as usize]);
            let end = range.offset() + range.len();
            *run = range.channel().range(start, end - start).ok_or_else(|| {
                io_error(STATUS, format!("DRAM channel {} run {start}..{end} leaves the channel", range.channel().index()))
            })?;
        }
        ttsim::dram_write(runs).map_err(|e| io_error(STATUS, e))
    }

    /// Read byte `off` of chunk `id` into `dst`. One DRAM read per channel; see
    /// [`Self::write_at`].
    pub fn read_at(&mut self, id: ChunkId, off: usize, dst: &mut [u8]) -> Result<(), BackendError> {
        const STATUS: ErrorStatus = ErrorStatus::MemoryCopyP2H;
        let addr = self.chunks[id].addr;
        let off = u64::try_from(off).map_err(|_| io_error(STATUS, "DRAM read offset does not fit u64".into()))?;
        let len = dst.len() as u64;
        self.check(id, off, len).map_err(|e| io_error(STATUS, e))?;
        let stripes = self.stripes(addr + off, len).map_err(|e| io_error(STATUS, e))?;
        let channel_count = usize::from(self.dram.channel_count());
        let mut runs: Vec<DramRange> = Vec::with_capacity(channel_count);
        let mut slot: Vec<Option<usize>> = vec![None; channel_count];
        for (range, _) in &stripes {
            let channel = usize::from(range.channel().index());
            match slot[channel] {
                Some(index) => {
                    let run = runs[index];
                    let start = run.offset();
                    let end = range.offset() + range.len();
                    runs[index] = run.channel().range(start, end - start).ok_or_else(|| {
                        io_error(STATUS, format!("DRAM channel {} run {start}..{end} leaves the channel", channel))
                    })?;
                }
                None => {
                    slot[channel] = Some(runs.len());
                    runs.push(*range);
                }
            }
        }
        let parts = ttsim::dram_read(runs).map_err(|e| io_error(STATUS, e))?;
        // Each channel's bytes arrive as one run; walk the pages in flat order
        // and take each page's slice from its channel's run.
        let mut cursor: Vec<usize> = vec![0; channel_count];
        for (range, at) in stripes {
            let channel = usize::from(range.channel().index());
            let index = slot[channel].expect("every striped channel has a run");
            let n = range.len() as usize;
            dst[at..at + n].copy_from_slice(&parts[index][cursor[channel]..cursor[channel] + n]);
            cursor[channel] += n;
        }
        Ok(())
    }

    fn check(&self, id: ChunkId, off: u64, len: u64) -> Result<(), String> {
        let chunk = &self.chunks[id];
        let end = off.checked_add(len).ok_or_else(|| format!("DRAM access at {off} of {len} bytes overflows"))?;
        if end > chunk.len {
            return Err(format!("DRAM access {off}..{end} of chunk {id:?} exceeds its {} bytes", chunk.len));
        }
        Ok(())
    }

    /// The per-channel DRAM ranges of chunk `id`'s bytes `off..off + len`, each
    /// paired with its byte offset inside that range.
    ///
    /// One range per [`STRIPE`]-byte page, so the range's bytes are contiguous
    /// in the L1 block that receives them: a mover entry carries one range and
    /// the same offset into the block. (Host transfers coalesce these back into
    /// one run per channel; a mover entry cannot, because its L1 side is
    /// contiguous only per page.)
    pub fn ranges(&self, id: ChunkId, off: usize, len: usize) -> Result<Vec<(DramRange, usize)>, String> {
        let chunk = self.chunks[id];
        let off = u64::try_from(off).map_err(|_| "DRAM access offset does not fit u64".to_string())?;
        let len = u64::try_from(len).map_err(|_| "DRAM access length does not fit u64".to_string())?;
        self.check(id, off, len)?;
        self.stripes(chunk.addr + off, len)
    }

    /// Split a flat DRAM range into the per-channel ranges it spans, each
    /// paired with its byte offset inside the caller's buffer.
    ///
    /// Page `p = addr / STRIPE` is on channel `p % channels` at channel offset
    /// `(p / channels) * STRIPE + addr % STRIPE`; accesses are cut at every page
    /// boundary, so a segment never crosses a channel. Round-robin at a fixed
    /// page size is what the card does; a different stripe is a change here and
    /// nowhere else.
    fn stripes(&self, addr: u64, len: u64) -> Result<Vec<(DramRange, usize)>, String> {
        let channels = u64::from(self.dram.channel_count());
        debug_assert!(channels > 0, "a DRAM grid with no usable channels");
        let mut out = Vec::with_capacity(len.div_ceil(STRIPE) as usize);
        let mut at = 0u64;
        while at < len {
            let flat = addr + at;
            let page = flat / STRIPE;
            let within = flat % STRIPE;
            let channel = self.dram.channel((page % channels) as u8).ok_or_else(|| {
                format!("flat DRAM address {flat:#x} names channel {}, which the chip does not have", page % channels)
            })?;
            let offset = (page / channels) * STRIPE + within;
            let take = (STRIPE - within).min(len - at);
            let range = channel.range(offset, take).ok_or_else(|| {
                format!(
                    "DRAM channel {} range {offset}..{} runs past the channel's {} bytes",
                    channel.index(),
                    offset + take,
                    tt_isa::dram::CHANNEL_BYTES
                )
            })?;
            out.push((range, at as usize));
            at += take;
        }
        Ok(out)
    }
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
    // The chip's DRAM channels. This is the simulator's grid (`Dram::FULL`,
    // every channel present): the native transport opens a simulated chip, and
    // the owner thread is the only holder of the device. Silicon has to take
    // the grid from `Device::dram_grid` instead — the ported layer refuses a
    // harvested chip (a p100's seven channels) until the translated DRAM
    // numbering is implemented.
    let dram = tt_isa::dram::Dram::FULL;
    // Flat extent: the card's GDDR6, but never more per channel than the NoC
    // can address — a flat space wider than `channels * CHANNEL_BYTES` would
    // strand its tail in a channel that does not exist.
    let dram_bytes = detect_dram_bytes().min(u64::from(dram.channel_count()) * tt_isa::dram::CHANNEL_BYTES);

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
                mma_variants: vec![],
                num_circular_buffers: 32, // architectural CB0-CB31
                has_openmp: false,
            },
            dram,
            total: dram_bytes,
            free_bytes: dram_bytes as Dim,
            chunks: Slab::new(),
            free_set: Set::default(),
            bump: 0,
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

/// One image: a single core's bytes, resident in device DRAM (NOTES §2,
/// "Table A"). Nothing on the card occupies two cores at once, so an image is
/// one `(core, address, len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Image {
    /// The core the bytes run on.
    core: Core,
    /// Flat device DRAM address of the image's first byte.
    address: u64,
    /// Byte length.
    len: u32,
    /// Identity: a recycled slab slot never matches a tile's ledger entry.
    generation: u32,
}

/// Slab key of an [`Image`] in the device's program cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ImageId(u32);

impl From<usize> for ImageId {
    fn from(value: usize) -> Self {
        ImageId(u32::try_from(value).expect("image id fits u32"))
    }
}

impl From<ImageId> for usize {
    fn from(value: ImageId) -> Self {
        value.0 as usize
    }
}

impl SlabId for ImageId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// Never-reused image identity, so a recycled [`ImageId`] is a ledger miss
/// rather than a stale hit.
static IMAGE_GENERATION: AtomicU32 = AtomicU32::new(1);

fn next_generation() -> u32 {
    IMAGE_GENERATION.fetch_add(1, AtomicOrdering::Relaxed)
}

/// The residency slot of a core's image (NOTES §2: B, NC, T0, T1, T2).
pub(crate) fn slot(core: Core) -> usize {
    match core {
        Core::B => 0,
        Core::NC => 1,
        Core::T0 => 2,
        Core::T1 => 3,
        Core::T2 => 4,
    }
}

/// The role thread an image's core feeds (`T0` -> 0).
fn role_thread(core: Core) -> u32 {
    match core {
        Core::T0 => 0,
        Core::T1 => 1,
        Core::T2 => 2,
        Core::B | Core::NC => todo!("{core:?} is a mover, not a role thread"),
    }
}

/// The L1 address a mover's image is loaded at (B at 0, NC at its image base,
/// which it reaches through the stub at its reset PC).
fn firmware_base(core: Core) -> u64 {
    match core {
        Core::B => tt_isa::dm::IMAGE_BASE,
        Core::NC => tt_isa::dm::nc::IMAGE_BASE,
        other => todo!("{other:?} has no mover image"),
    }
}

#[derive(Debug)]
struct TTProgram {
    /// The legs this kernel runs, in program order (NOTES §2, "Table B"): the
    /// T0/T1/T2 images of a compute kernel. Reader and writer images join the
    /// sequence once the `Dm` stream is compilation output.
    sequence: Box<[ImageId]>,
    /// The program length word of each leg, parallel to `sequence`: what the
    /// role's descriptor names (`code::stored`, loop header included).
    lens: Box<[u32]>,
    /// The DRAM boundary of the run, in program order, as the rendered
    /// kernel's `DmRead`/`DmWrite` pages. This is the backend's own table:
    /// the launch substitutes a chunk address for each page's param and adds
    /// the grid point's offsets, deciding nothing (NOTES §2).
    reader: Box<[TTDmMove]>,
    writer: Box<[TTDmMove]>,
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
    /// The device's program cache: every image on the card (NOTES §2).
    images: Slab<ImageId, Image>,
    /// The two mover firmware images ([`tt_firmware_images::DM_B`],
    /// [`DM_NC`]), created on first compile: boot is their first residency
    /// miss, not a special step.
    firmware: Option<[ImageId; 2]>,
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
        // A rendered kernel is the finished stream (NOTES §2): this walk only
        // collects what the launch needs — the grid, the role programs, the
        // boundary pages and the semaphore set. Nothing here decides anything.
        let rendered = kernel.render()?;
        let mut reader: Vec<TTDmMove> = Vec::new();
        let mut writer: Vec<TTDmMove> = Vec::new();
        let mut roles: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut math_loops: Vec<[u32; 3]> = Vec::new();
        let mut semset: Vec<TTSemaphoreInit> = Vec::new();
        let mut gws: Option<Vec<GwsDim>> = None;
        let mut n_inputs = 0u32;
        let mut n_outputs = 0u32;
        let mut sources: [String; 3] = [String::new(), String::new(), String::new()];
        let mut section = 0usize;
        let mut op_id = rendered.head;
        while !op_id.is_null() {
            match rendered.at(op_id) {
                Op::GPU(gpu) => {
                    let GPUOp::Params(kinds) = &**gpu else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("{op_id:?} is not a Params op").into(),
                        });
                    };
                    for kind in kinds.iter() {
                        match kind {
                            ParamKind::Global => n_inputs += 1,
                            ParamKind::GlobalMut => n_outputs += 1,
                            ParamKind::Variable => {}
                        }
                    }
                }
                Op::TT(TTOp::TensixGrid(grid)) => gws = Some(grid.to_vec()),
                Op::TT(TTOp::Unpack(words)) => roles[0] = words.to_vec(),
                Op::TT(TTOp::Math(program)) => {
                    roles[1] = program.words.to_vec();
                    math_loops = program.loops.to_vec();
                }
                Op::TT(TTOp::Pack(words)) => roles[2] = words.to_vec(),
                Op::TT(TTOp::DmRead(page)) => reader.push(TTDmMove {
                    param: page.param,
                    l1: page.l1,
                    bytes: page.len,
                    narrow: TT_NARROW_NONE,
                    base: page.base,
                    gstride0: page.gstride0,
                    gstride1: page.gstride1,
                }),
                // One whole tile image, shaped by the mover: `READ_BROADCAST_COL`
                // copies the header and writes row `r`'s value into every column.
                Op::TT(TTOp::DmReadBroadcastCol(tile)) => reader.push(TTDmMove {
                    param: tile.param,
                    l1: tile.dst,
                    bytes: tt_isa::dm::TILE_SLOT as u32,
                    narrow: TT_NARROW_NONE,
                    base: tile.base,
                    gstride0: tile.gstride0,
                    gstride1: tile.gstride1,
                }),
                Op::TT(TTOp::DmWrite(page)) => writer.push(TTDmMove {
                    param: page.param,
                    l1: page.l1,
                    bytes: page.len,
                    narrow: page.narrow,
                    base: page.base,
                    gstride0: page.gstride0,
                    gstride1: page.gstride1,
                }),
                Op::TT(TTOp::Semset { sem, value }) => {
                    semset.push(TTSemaphoreInit { index: *sem, value: *value, max: tt_isa::sync::MAX_VALUE })
                }
                Op::Source(src) => sources[section] = src.as_str().to_string(),
                Op::TT(TTOp::EndReader) => section = 1,
                Op::TT(TTOp::EndCompute) => section = 2,
                other => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("{op_id:?} is not a rendered TT op: {other:?}").into(),
                    });
                }
            }
            op_id = rendered.next_op(op_id);
        }
        let Some(gws) = gws else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "rendered kernel has no TensixGrid".into(),
            });
        };
        if debug_asm {
            eprintln!("[tenstorrent] reader:\n{}", sources[0]);
            eprintln!("[tenstorrent] compute:\n{}", sources[1]);
            eprintln!("[tenstorrent] writer:\n{}", sources[2]);
            for (role, words) in roles.iter().enumerate() {
                eprintln!("[tenstorrent] role T{role}: {} words", words.len());
            }
            eprintln!("[tenstorrent] dm: {} reader pages, {} writer pages", reader.len(), writer.len());
        }

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

        // NOTES §2: codegen's bytes become images in the device's program
        // cache, uploaded into DRAM once. The stored form is what a program
        // slot holds, so its length word is decided here and never at launch.
        self.ensure_firmware()?;
        let mut sequence: Vec<ImageId> = Vec::with_capacity(3);
        let mut lens: Vec<u32> = Vec::with_capacity(3);
        for (thread, words) in roles.iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            // Block repeats belong to T1's stream (row loops are SFPU work).
            let loops: Vec<kern::code::Loop> = if thread == 1 {
                math_loops.iter().map(|l| kern::code::Loop { start: l[0], len: l[1], count: l[2] }).collect()
            } else {
                Vec::new()
            };
            let (stored, len) = kern::code::stored(words, &loops).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent: role T{thread} program: {e}").into(),
            })?;
            let bytes: Vec<u8> = stored.iter().flat_map(|w| w.to_le_bytes()).collect();
            sequence.push(self.upload([Core::T0, Core::T1, Core::T2][thread], &bytes)?);
            lens.push(len);
        }
        let mut sems: Vec<SemaphoreInit> = Vec::with_capacity(semset.len());
        for s in &semset {
            let sem = tt_isa::sync::Semaphore::new(s.index).ok_or_else(|| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("semaphore {} is outside 0..8", s.index).into(),
            })?;
            sems.push((sem, s.value, s.max));
        }
        Ok(self.programs.push(TTProgram {
            sequence: sequence.into_boxed_slice(),
            lens: lens.into_boxed_slice(),
            reader: reader.into_boxed_slice(),
            writer: writer.into_boxed_slice(),
            sems,
            n_inputs,
            n_outputs,
            gws,
            max_grid,
        }))
    }

    /// Put `bytes` into device DRAM and register them as the image of `core`.
    /// This is the one PCIe write a compiled program costs; every later load
    /// of it is a NoC read from this copy.
    fn upload(&mut self, core: Core, bytes: &[u8]) -> Result<ImageId, BackendError> {
        let Pool::TT(dev_id) = self.memory_pool else {
            unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
        };
        let tpool = pool(dev_id)?;
        let mut tpool = super::lock(self.memory_pool, tpool);
        let chunk = tpool.allocate(bytes.len() as Dim)?;
        tpool.write_at(chunk, 0, bytes)?;
        let image = Image {
            core,
            address: tpool.chunks[chunk].addr,
            len: u32::try_from(bytes.len()).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent: {} bytes do not fit an image", bytes.len()).into(),
            })?,
            generation: next_generation(),
        };
        Ok(self.images.push(image))
    }

    /// The two mover firmware images, created once: the backend's own leaves,
    /// not derived from any `Kernel` (NOTES §2).
    fn ensure_firmware(&mut self) -> Result<[ImageId; 2], BackendError> {
        if let Some(ids) = self.firmware {
            return Ok(ids);
        }
        let ids = [
            self.upload(Core::B, tt_firmware_images::DM_B.1)?,
            self.upload(Core::NC, tt_firmware_images::DM_NC.1)?,
        ];
        self.firmware = Some(ids);
        Ok(ids)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        let Some(program) = self.programs.get(program_id) else {
            return;
        };
        let sequence = program.sequence.clone();
        self.programs.remove(program_id);
        // Autotune is the one caller: drop the images it compiled and give
        // their DRAM back. Every other program is never released.
        let Pool::TT(dev_id) = self.memory_pool else {
            unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
        };
        let Ok(tpool) = pool(dev_id) else { return };
        let mut tpool = super::lock(self.memory_pool, tpool);
        for id in sequence.iter() {
            let Some(image) = self.images.get(*id).copied() else {
                continue;
            };
            self.images.remove(*id);
            let chunk = tpool.chunks.iter().find(|(_, c)| c.addr == image.address).map(|(chunk, _)| chunk);
            if let Some(chunk) = chunk {
                tpool.release(chunk);
            }
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
        // Grid dispatch runs one program per grid point below; the extents
        // must name real tiles.
        if grid_dims[0] == 0 || grid_dims[1] == 0 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tenstorrent grid of {} x {} has no cores", grid_dims[0], grid_dims[1]).into(),
            });
        }
        // The boundary: reader moves bring the param's DRAM bytes into the CB's
        // L1 address before any role starts; writer moves write them back out
        // after every role finished. Both are the tile's data movers' lists
        // (`ttsim`), not host copies, so no role touches DRAM and no tensor
        // byte crosses the host inside a launch. A 16-bit output is the one
        // exception: its narrowing is the host's (see the writer loop below),
        // so that move is read back instead.
        //
        // Grid dispatch: one run per grid point, on that point's Tensix tile,
        // with each boundary move offset by the point's group indices. The
        // program is tile-uniform (same L1 layout, same instruction streams);
        // the DRAM slice is the only thing that differs, and `TTDmMove`'s
        // `base`/`gstride*` are what carry it.
        let Pool::TT(dev_id) = self.memory_pool else {
            unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
        };
        let tpool = pool(dev_id)?;
        use tt_isa::noc::Noc0;
        use tt_isa::noc::grid::Tensix;
        let tensix = Tensix::FULL;
        let ncols = tensix.columns().count() as u32;
        let tiles: Vec<(u8, u8)> = tensix.tiles::<Noc0>().map(|t| (t.x(), t.y())).collect();
        // The kernel's images, resolved once: which slot each leg fills and the
        // DRAM pages that fill it. Loading is the owner thread's decision (the
        // residency ledger); everything here is address translation (NOTES §2).
        let firmware = self.firmware.ok_or_else(|| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: "tenstorrent launch before any compile created the mover images".into(),
        })?;
        let mut programs: [Option<ttsim::RoleProgram>; 3] = [None, None, None];
        let firmware: [ttsim::Firmware; 2] = {
            let tpool = super::lock(self.memory_pool, tpool);
            for (leg, id) in prog.sequence.iter().enumerate() {
                let image = self.images[*id];
                let thread = role_thread(image.core);
                let slot_at = Mailbox::of(thread).program();
                let ranges = tpool.stripes(image.address, u64::from(image.len)).map_err(|e| BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("program image {id:?}: {e}").into(),
                })?;
                programs[thread as usize] = Some(ttsim::RoleProgram {
                    id: *id,
                    generation: image.generation,
                    len: prog.lens[leg],
                    ranges: ranges.into_iter().map(|(range, at)| (range, (slot_at + at as u64) as u32)).collect(),
                });
            }
            let mut fw = Vec::with_capacity(2);
            for id in firmware {
                let image = self.images[id];
                let ranges = tpool.stripes(image.address, u64::from(image.len)).map_err(|e| BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("firmware image {id:?}: {e}").into(),
                })?;
                fw.push(ttsim::Firmware {
                    id,
                    generation: image.generation,
                    core: image.core,
                    base: firmware_base(image.core),
                    ranges: ranges.into_iter().map(|(range, _)| range).collect(),
                });
            }
            let nc = fw.pop().expect("the firmware images are created together");
            let b = fw.pop().expect("the firmware images are created together");
            [b, nc]
        };
        for gidx0 in 0..grid_dims[0] {
            for gidx1 in 0..grid_dims[1] {
                let point = usize::try_from(gidx0 * ncols + gidx1).unwrap_or(usize::MAX);
                let (x, y) = *tiles.get(point).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tenstorrent grid point ({gidx0}, {gidx1}) is not a Tensix tile").into(),
                })?;
                let mut reads: Vec<(DramRange, u32)> = Vec::new();
                let mut writes: Vec<(DramRange, u32)> = Vec::new();
                let mut narrows: Vec<(u32, ChunkId, usize, u64, usize, u8)> = Vec::new();
                {
                    let tpool = super::lock(self.memory_pool, tpool);
                    for mv in prog.reader.iter() {
                        let chunk = chunks[mv.param as usize].ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: format!("reader move names param {}, which has no buffer arg", mv.param).into(),
                        })?;
                        let ranges = tpool.ranges(chunk, tt_move_offset(mv, gidx0, gidx1), mv.bytes as usize).map_err(|e| {
                            BackendError {
                                status: ErrorStatus::KernelLaunch,
                                context: format!("reader move of param {}: {e}", mv.param).into(),
                            }
                        })?;
                        for (range, at) in ranges {
                            reads.push((range, mv.l1 + at as u32));
                        }
                    }
                    for mv in prog.writer.iter() {
                        let chunk = chunks[mv.param as usize].ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: format!("writer move names param {}, which has no buffer arg", mv.param).into(),
                        })?;
                        let off = tt_move_offset(mv, gidx0, gidx1);
                        let ranges = tpool.ranges(chunk, off, mv.bytes as usize).map_err(|e| BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: format!("writer move of param {}: {e}", mv.param).into(),
                        })?;
                        if mv.narrow == TT_NARROW_NONE {
                            for (range, at) in ranges {
                                writes.push((range, mv.l1 + at as u32));
                            }
                        } else {
                            // A narrowed output cannot be written by the mover:
                            // the conversion is the host's (see below).
                            narrows.push((mv.param, chunk, off, u64::from(mv.l1), mv.bytes as usize, mv.narrow));
                        }
                    }
                }
                let read_back: Vec<(u64, usize)> = narrows.iter().map(|(_, _, _, l1, bytes, _)| (*l1, *bytes)).collect();
                let run = NativeRun {
                    tile: (x, y),
                    programs: programs.clone(),
                    firmware: firmware.clone(),
                    reads,
                    writes,
                    read_back,
                    sems: prog.sems.clone(),
                };
                let outcome = run_native(run)
                    .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: e.into_boxed_str() })?;
                let mut tpool = super::lock(self.memory_pool, tpool);
                for (idx, (param, chunk, off, _, bytes, narrow)) in narrows.iter().enumerate() {
                    let out = &outcome.l1[idx];
                    // Writer-side narrowing happens on the host, on the read-back
                    // bytes: the packer cannot do it (F32 `Dst` -> BF16 L1 is
                    // `pack_fmt_conv_mode=0x105`, which the pinned ttsim refuses).
                    let narrowed: Vec<u8> = match *narrow {
                        TT_NARROW_BF16 => {
                            debug_assert_eq!(out.len() % 4, 0, "narrowing a partial f32 datum");
                            let mut bytes_out = Vec::with_capacity(out.len() / 2);
                            for datum in out.chunks_exact(4) {
                                let d = f32::from_ne_bytes(datum.try_into().expect("four bytes"));
                                bytes_out.extend_from_slice(&bf16::from_f32(d).to_bits().to_ne_bytes());
                            }
                            bytes_out
                        }
                        TT_NARROW_F16 => {
                            debug_assert_eq!(out.len() % 4, 0, "narrowing a partial f32 datum");
                            let mut bytes_out = Vec::with_capacity(out.len() / 2);
                            for datum in out.chunks_exact(4) {
                                let d = f32::from_ne_bytes(datum.try_into().expect("four bytes"));
                                bytes_out.extend_from_slice(&f16::from_f32(d).to_bits().to_ne_bytes());
                            }
                            bytes_out
                        }
                        TT_NARROW_NONE => unreachable!("writer param {param} is 8-bit wide: nothing to narrow"),
                        other => unreachable!("writer narrowing code {other} is not TT_NARROW_F16/BF16"),
                    };
                    debug_assert_eq!(narrowed.len(), bytes / 2, "writer narrowing changed the datum count");
                    tpool.write_at(*chunk, *off, &narrowed)?;
                }
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
    /// The destination is device DRAM, so the move is a download from the
    /// source's own pool into a host staging buffer, then an upload into the
    /// chunk. Single-shard placements only.
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
        let Pool::TT(dev_id) = self.memory_pool else {
            unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
        };
        let mut tpool = super::lock(self.memory_pool, pool(dev_id)?);
        tpool.host_to_pool(&buf, dst_shard.chunk)
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
    pub(crate) fn tilize_transfer(&self, src: &Arc<Placement>, shape: &[Dim], dtype: DType) -> Result<Arc<Placement>, ZyxError> {
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
        let (pr, pc) = self.info().expect("tilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        // Device tiles are tile *images*: a 16-byte header the unpacker steps
        // over (`UNPACR_Regular`'s `(Base + 1 + DigestSize) * 16`), the 1024
        // face-order datums, and padding to 64 bytes (`tt_isa::dm::TILE_SLOT`).
        // The header is never read; the movers copy it along so what lands in
        // L1 is a valid tile image (the shaped reads need it, see `STRIPE`).
        let el_d = el as Dim;
        let slot = tt_isa::dm::TILE_SLOT as Dim;
        let hdr = tt_isa::dm::TILE_DATA as Dim;
        let tile_elems = slot / el_d;
        let hdr_elems = hdr / el_d;
        let tiles = (pr * pc) / 1024;
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(tiles * tile_elems);
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
        let mut physical = vec![0u8; usize::try_from(outer * tiles * slot).expect("TT tile image bytes fit usize")];
        for b in 0..outer {
            for r in 0..rows {
                for c in 0..cols {
                    let (tr, tc) = (r / 32, c / 32);
                    let face = (r % 32) / 16 * 2 + (c % 32) / 16;
                    let within = (r % 16) * 16 + (c % 16);
                    let srci = ((b * rows + r) * cols + c) * el_d;
                    let dsti = (((b * ntr + tr) * ntc + tc) * tile_elems + hdr_elems + face * 256 + within) * el_d;
                    physical[dsti as usize..dsti as usize + el].copy_from_slice(&logical[srci as usize..srci as usize + el]);
                }
            }
        }
        // Physical bytes to this device.
        let plen = physical.len();
        let chunk = Pool::Host.insert_host(physical.into_boxed_slice());
        let staged = Arc::new(Placement { shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: plen }] });
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
        let (pr, pc) = self.info().expect("untilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        // The same tile images [`Dev::tilize_transfer`] writes (see there and
        // `STRIPE`): header + face-order datums, `tt_isa::dm::TILE_SLOT` each.
        let el_d = el as Dim;
        let slot = tt_isa::dm::TILE_SLOT as Dim;
        let hdr = tt_isa::dm::TILE_DATA as Dim;
        let tile_elems = slot / el_d;
        let hdr_elems = hdr / el_d;
        let tiles = (pr * pc) / 1024;
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(tiles * tile_elems);
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
        let physical_len = usize::try_from(outer * tiles * slot).expect("TT tile image bytes fit usize");
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
                    let srci = (((b * ntr + tr) * ntc + tc) * tile_elems + hdr_elems + face * 256 + within) * el_d;
                    let dsti = ((b * rows + r) * cols + c) * el as Dim;
                    logical[dsti as usize..dsti as usize + el].copy_from_slice(&physical[srci as usize..srci as usize + el]);
                }
            }
        }
        let logical_len = logical.len();
        let chunk = Pool::Host.insert_host(logical.into_boxed_slice());
        let staged = Arc::new(Placement { shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: logical_len }] });
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
