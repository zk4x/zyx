// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Second-generation backend command queue and scheduler.
//!
//! Design (`backend/hcq.md`):
//! - The queue holds [`Cmd`]s only: [`Cmd::Launch`] and [`Cmd::Copy`].
//!   Host/disk ingestion creates [`Placement`]s directly, never via queue.
//! - `OpId`s are per-graph positional slots, stable within their graph,
//!   rebound with fresh placements per replay. Uniqueness scope is the
//!   producing graph; namespaces never meet across batches.
//! - [`Placement`]s are `Arc`-owned boundary values. Interior regions are
//!   scheduler-private.
//! - Arenas live behind one global static (own lock; `RT -> arena` order,
//!   never reversed; arena code never calls back into the runtime).
//! - Execution is async with GPU-side serialization: each device runs one
//!   graph at a time (same stream or event chaining — a backend
//!   requirement), the CPU runs ahead. A launch records a completion event;
//!   regions are freed only behind event completion (sweep at compile
//!   start, non-blocking). Nothing is ever freed from under in-flight work.
//! - Lifetime is explicit handoffs, all under the RT lock: leaf destruction
//!   and previous-output replacement free behind the device's last event
//!   (immediately if nothing ever launched); [`ExecutionGraph::dispose`]
//!   synchronizes first (the one allowed sync, off the hot path).
//! - v1 constraints (lifted by backend ports): batches are single-device
//!   (multi-device is a loud failure).
//! - [`ExecutionGraph`] holds backend-native executables only, plus
//!   allocator metadata. Backends consume resolved [`Partition`]s at
//!   compile and own rebinding natively.
//!
//! Status: backend-agnostic pipeline is real; backend seams (`todo!()`:
//! chunk growth, program metadata, executable build/launch/events) fill in
//! per backend port.

#![allow(dead_code)]

use crate::{Map, kernel::{OpId, ParamKind}};
use super::{Dev, ProgramId};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// One device-resident (or host-resident) piece of a placed value.
#[derive(Debug, Clone)]
pub enum Shard {
    /// Host-owned bytes (loaded data, weights staging, readback targets).
    Host {
        data: Vec<u8>,
    },
    /// A region inside a scheduler arena on `device`.
    Device {
        device: Dev,
        ptr: u64,
        size: u64,
    },
}

/// A placed value: one [`Shard`] per device holding it (one shard for the
/// common single-device case, several for sharded tensors).
#[derive(Debug, Clone)]
pub struct Placement {
    pub shards: Vec<Shard>,
}

/// Scheduler commands. Args and outputs are batch-local [`OpId`] slots; the
/// scheduler resolves boundary slots through the per-compile boundary table
/// and interior slots positionally from def-use within the batch.
#[derive(Debug)]
pub enum Cmd {
    /// Run `program`. Devices come from `program.dev`; arg regions must
    /// agree (cross-device staging of one value into one node is later work).
    Launch {
        program: ProgramId,
        args: Vec<OpId>,
        outputs: Vec<OpId>,
    },
    /// Copy from `src` value to `dst` value. Fresh `dst`s land on the source
    /// device; device-crossing copies target boundary values.
    Copy {
        src: OpId,
        dst: OpId,
    },
}

impl Cmd {
    /// Value reads: launch args plus copy sources. Outputs excluded.
    fn reads(&self) -> Vec<OpId> {
        match self {
            Cmd::Launch { args, .. } => args.clone(),
            Cmd::Copy { src, .. } => vec![*src],
        }
    }

    /// Value defs: launch outputs plus copy destinations.
    fn defs(&self) -> Vec<OpId> {
        match self {
            Cmd::Launch { outputs, .. } => outputs.clone(),
            Cmd::Copy { dst, .. } => vec![*dst],
        }
    }
}

/// Ordered command batch. Eager builds one per kernel; plan builds one per
/// graph. Drained by swap at flush; every access happens under the
/// `Runtime` lock, so no second lock lives here.
#[derive(Debug, Default)]
pub struct CmdQueue {
    cmds: Vec<Cmd>,
}

impl CmdQueue {
    /// Append one command, in program order.
    pub fn push(&mut self, cmd: Cmd) {
        self.cmds.push(cmd);
    }

    /// Swap-drain the batch for compilation.
    pub fn take_batch(&mut self) -> Vec<Cmd> {
        std::mem::take(&mut self.cmds)
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.cmds.len()
    }

    /// True when no commands are queued.
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }
}

/// One append-only arena backing scratch on a single device. Chunk bases
/// never move, so frozen addresses stay valid for the process lifetime.
#[derive(Debug)]
struct Arena {
    device: Dev,
    /// (base, size) per chunk. Bases come from backend allocators and never change.
    chunks: Vec<(u64, u64)>,
    /// Bump cursor (bytes consumed) per chunk.
    used: Vec<u64>,
    /// Free regions as (chunk, offset, size, source op). The source is the
    /// dead op whose death freed the region (`None` for explicitly returned
    /// memory, covered by completion gating); reuse edges the new def after
    /// the source's last touch. Coalescing merges same-source neighbours
    /// only, never across sources.
    free: Vec<(u32, u64, u64, Option<OpId>)>,
}

/// Alignment (bytes) for every arena placement.
const ARENA_ALIGN: u64 = 256;

const fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) / align * align
}

impl Arena {
    fn new(device: Dev) -> Self {
        Self { device, chunks: Vec::new(), used: Vec::new(), free: Vec::new() }
    }

    /// Register one backend-allocated chunk. Bases must never move or be
    /// freed while any graph lives; the backend guarantees this.
    fn register_chunk(&mut self, base: u64, size: u64) {
        assert!(size > 0, "mod2: cannot register an empty chunk");
        self.chunks.push((base, size));
        self.used.push(0);
    }

    /// Reverse lookup: which (chunk, offset) holds `[ptr, ptr + size)?
    fn find(&self, ptr: u64, size: u64) -> Option<(u32, u64)> {
        for (i, (base, len)) in self.chunks.iter().enumerate() {
            if *base <= ptr && ptr + size <= *base + *len {
                return u32::try_from(i).ok().map(|chunk| (chunk, ptr - *base));
            }
        }
        None
    }

    /// Best-fit from the free list, else bump. Returns (chunk, offset, ptr,
    /// source op). Bump allocations carry no source (untouched memory).
    fn alloc(&mut self, size: u64) -> Option<(u32, u64, u64, Option<OpId>)> {
        let size = align_up(size, ARENA_ALIGN);
        let mut best: Option<(usize, u32, u64)> = None;
        for (i, (chunk, off, len, _)) in self.free.iter().enumerate() {
            let start = align_up(*off, ARENA_ALIGN);
            if start + size <= *off + *len {
                let leftover = (*off + *len) - (start + size);
                if best.is_none_or(|(_, _, l)| leftover < l) {
                    best = Some((i, *chunk, start));
                }
            }
        }
        if let Some((i, chunk, start)) = best {
            let (_, off, len, source) = self.free.swap_remove(i);
            let end = off + len;
            if start > off {
                self.free.push((chunk, off, start - off, source));
            }
            if start + size < end {
                self.free.push((chunk, start + size, end - (start + size), source));
            }
            let base = self.chunks[chunk as usize].0;
            return Some((chunk, start, base + start, source));
        }
        for (c, ((base, len), used)) in self.chunks.iter().zip(self.used.iter_mut()).enumerate() {
            let start = align_up(*used, ARENA_ALIGN);
            if start + size <= *len {
                *used = start + size;
                let chunk = u32::try_from(c).expect("mod2: chunk index overflow");
                return Some((chunk, start, *base + start, None));
            }
        }
        None
    }

    /// Return a region from dead op `source` (`None`: covered by completion
    /// gating, no reuse edge). Coalesce same-source neighbours only.
    fn release(&mut self, chunk: u32, offset: u64, size: u64, source: Option<OpId>) {
        self.free.push((chunk, offset, size, source));
        self.free.sort();
        let mut merged: Vec<(u32, u64, u64, Option<OpId>)> = Vec::with_capacity(self.free.len());
        for (c, o, s, src) in self.free.drain(..) {
            if let Some((lc, lo, ls, lsrc)) = merged.last_mut() {
                if *lc == c && *lsrc == src && *lo + *ls == o {
                    *ls += s;
                    continue;
                }
            }
            merged.push((c, o, s, src));
        }
        self.free = merged;
    }
}

/// One in-flight launch: completion event plus everything that must survive
/// until it (executable, interior regions, retired placements). Freed and
/// destroyed by the sweep once the event completes — never before.
#[derive(Debug)]
struct PendingLaunch {
    device: Dev,
    event: u64,
    regions: Vec<(u32, u64, u64)>,
    exec: Option<BackendExec>,
    prev: Vec<Arc<Placement>>,
}

/// All device arenas plus in-flight launches and last-launch events, behind
/// the one global static. Lock order is `RT -> arena`, never reversed;
/// arena code never calls back into the runtime.
#[derive(Debug, Default)]
struct Arenas {
    vec: Vec<Arena>,
    pending: Vec<PendingLaunch>,
    last_event: Vec<(Dev, u64)>,
}

static ARENAS: Mutex<Arenas> = Mutex::new(Arenas { vec: Vec::new(), pending: Vec::new(), last_event: Vec::new() });

/// One resolved value region. `slot` is the arena position for interior
/// regions (`None` for boundary regions, which free via reverse lookup).
#[derive(Debug, Clone, Copy)]
struct Region {
    device: Dev,
    ptr: u64,
    size: u64,
    slot: Option<(u32, u64)>,
}

/// Template reference to a value inside one compiled batch. Fixed regions
/// are frozen at compile; rebind values bind per replay from fresh
/// placements (boundary inputs and outputs alike).
#[derive(Debug, Clone)]
enum ArgRef {
    Fixed(Vec<Region>),
    Rebind(OpId),
    Scalar(i64),
}

/// One resolved op handed to a backend: a kernel launch (`program` set) or a
/// copy (`program` none, single source arg, single destination output).
/// Compile-time handoff data only, consumed immediately by executable build.
#[derive(Debug)]
struct PartitionOp {
    program: Option<ProgramId>,
    device: Dev,
    args: Vec<ArgRef>,
    outputs: Vec<ArgRef>,
}

/// One device's share of a batch: resolved ops plus dependency edges (node
/// indices local to `ops`). v1 batches are single-device; multi-device is a
/// loud failure until the backend ports restore it with native event sync.
#[derive(Debug)]
struct Partition {
    device: Dev,
    ops: Vec<PartitionOp>,
    edges: Vec<(usize, usize)>,
}

/// Backend-native executables: one compiled whole per device partition.
/// A CUDA graph, a C table, a Vulkan pipeline set — each backend fills its
/// own arm when it ports over. Stubs until then.
#[derive(Debug)]
pub enum BackendExec {
    Cuda(CudaExec),
    C(CExec),
    Vulkan(VulkanExec),
    OpenCL(OpenCLExec),
}

/// CUDA graph executable (stub: filled by the CUDA port).
#[derive(Debug)]
pub struct CudaExec;

/// C backend executable (stub: filled by the C port).
#[derive(Debug)]
pub struct CExec;

/// Vulkan pipeline set (stub: filled by the Vulkan port).
#[derive(Debug)]
pub struct VulkanExec;

/// OpenCL executable (stub: filled by the OpenCL port).
#[derive(Debug)]
pub struct OpenCLExec;

/// One exported output: slot id, size, device. Regions are assigned per
/// replay (reused from the previous output iff its `Arc` is otherwise dead).
#[derive(Debug)]
struct OutputSpec {
    op: OpId,
    size: u64,
    device: Dev,
}

/// Backend-agnostic scheduler. Stateless across compiles: boundary values
/// arrive per compile in the caller boundary table, arenas live behind the
/// global static. Interior ids resolve positionally within their batch; a
/// def of a boundary-table op is in-place mutation (assign): regions are
/// reused, sizes must match. A duplicate interior def panics (not SSA).
/// Defs unused within the batch are exported under their own ids, with
/// regions assigned at first replay.
#[derive(Debug, Default)]
pub struct Scheduler {
    _sealed: (),
}

impl Scheduler {
    /// Scheduler handle (all state lives behind the global arenas static).
    pub fn new() -> Self {
        Self { _sealed: () }
    }

    /// Register one backend-allocated chunk on `device`.
    pub fn register_chunk(&self, device: Dev, base: u64, size: u64) {
        let mut guard = Self::arenas();
        let arenas: &mut Arenas = &mut guard;
        Self::arena_for(arenas, device).register_chunk(base, size);
    }

    /// Explicit free of one device region (leaf-destruction / dispose
    /// handoffs, after the caller verified sole ownership). Gated behind
    /// the device's last launch event: immediate iff nothing ever launched.
    pub fn free_region(device: Dev, ptr: u64, size: u64) {
        let mut guard = Self::arenas();
        let arenas: &mut Arenas = &mut guard;
        match Self::last_event(arenas, device) {
            None => {
                let (chunk, offset) = Self::find_region(arenas, device, ptr, size);
                Self::arena_for(arenas, device).release(chunk, offset, size, None);
            }
            Some(event) => arenas.pending.push(PendingLaunch {
                device,
                event,
                regions: vec![Self::locate_region(arenas, device, ptr, size)],
                exec: None,
                prev: Vec::new(),
            }),
        }
    }

    fn arenas() -> std::sync::MutexGuard<'static, Arenas> {
        ARENAS.lock().expect("mod2: arenas lock poisoned by a panicking holder")
    }

    fn arena_for(arenas: &mut Arenas, device: Dev) -> &mut Arena {
        if let Some(i) = arenas.vec.iter().position(|a| a.device == device) {
            return &mut arenas.vec[i];
        }
        arenas.vec.push(Arena::new(device));
        arenas.vec.last_mut().expect("mod2: arena just pushed")
    }

    fn find_region(arenas: &Arenas, device: Dev, ptr: u64, size: u64) -> (u32, u64) {
        arenas
            .vec
            .iter()
            .find(|a| a.device == device)
            .expect("mod2: region on a device without an arena")
            .find(ptr, size)
            .expect("mod2: region is outside all arena chunks")
    }

    fn locate_region(arenas: &Arenas, device: Dev, ptr: u64, size: u64) -> (u32, u64, u64) {
        let (chunk, offset) = Self::find_region(arenas, device, ptr, size);
        (chunk, offset, size)
    }

    fn last_event(arenas: &Arenas, device: Dev) -> Option<u64> {
        arenas.last_event.iter().find(|(d, _)| *d == device).map(|(_, e)| *e)
    }

    /// Backend seam: allocate and register one chunk of at least `min_size`
    /// bytes on `device`.
    fn grow(_device: Dev, _min_size: u64) {
        todo!("mod2 backend seam: allocate arena chunk on device");
    }

    /// Backend seam: byte size of the `index`-th output of `program`.
    /// Backends record this at kernel compile from the kernel IR shapes.
    fn output_bytes(_program: ProgramId, _index: usize) -> u64 {
        todo!("mod2 backend seam: program output sizes");
    }

    /// Backend seam: flat param kinds of `program` in launch-arg order.
    fn program_params(_program: ProgramId) -> Vec<ParamKind> {
        todo!("mod2 backend seam: program param kinds");
    }

    /// Backend seam: concrete value of a scalar (const/variable) op.
    fn scalar_value(_op: OpId) -> i64 {
        todo!("mod2 backend seam: scalar op evaluation");
    }

    /// Backend seam: lower one partition to its native executable,
    /// translating edges to native sync.
    fn build_exec(partition: Partition) -> BackendExec {
        match partition.device {
            Dev::Cuda(_) => todo!("mod2: CUDA port — lower partition to cudaGraphExec"),
            Dev::C => todo!("mod2: C port"),
            Dev::Vulkan(_) => todo!("mod2: Vulkan port"),
            Dev::OpenCL(_) => todo!("mod2: OpenCL port"),
            Dev::Cblas => todo!("mod2: CBLAS port"),
            Dev::Dummy => todo!("mod2: dummy port"),
            Dev::Auto => panic!("mod2: partition on Dev::Auto (resolve before compile)"),
            #[cfg(feature = "tenstorrent")]
            Dev::TT(_) => todo!("mod2: tenstorrent port"),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(_) => todo!("mod2: wgpu port"),
        }
    }

    /// Backend seam: rebind one executable's boundary nodes from fresh
    /// regions and launch it asynchronously on the device's serial stream.
    /// Returns the completion event (signaled after the launch finishes).
    fn launch_exec(_exec: &BackendExec, _bindings: &Map<OpId, Vec<Region>>) -> u64 {
        todo!("mod2 backend seam: rebind, async launch, return completion event");
    }

    /// Backend seam: non-blocking completion query. Never blocks the CPU.
    fn event_done(_event: u64) -> bool {
        todo!("mod2 backend seam: non-blocking event query");
    }

    /// Backend seam: block until all work on `device` completes. Only used
    /// by explicit dispose (off the hot path), never by replay or compile.
    fn sync_device(_device: Dev) {
        todo!("mod2 backend seam: blocking device synchronize");
    }

    /// Destroy a finished executable's native resources (no-op for current
    /// stubs; backends fill in at port time).
    fn destroy_exec(_exec: BackendExec) {}

    /// Sweep completed launches: non-blocking event queries only, so the
    /// CPU never waits. Completed launches release their regions and drop
    /// their executables. Call at compile start to bound memory while the
    /// CPU runs ahead of the device.
    fn sweep(arenas: &mut Arenas) {
        let mut i = 0;
        while i < arenas.pending.len() {
            if Self::event_done(arenas.pending[i].event) {
                let launch = arenas.pending.swap_remove(i);
                for (chunk, offset, size) in launch.regions {
                    Self::arena_for(arenas, launch.device).release(chunk, offset, size, None);
                }
                if let Some(exec) = launch.exec {
                    Self::destroy_exec(exec);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Regions of one placement's shards. Host shards panic: device kernels
    /// only ever see device regions (feed device input regions first).
    fn regions_of(placement: &Placement, op: OpId) -> Vec<Region> {
        let mut regions = Vec::with_capacity(placement.shards.len());
        for shard in &placement.shards {
            match shard {
                Shard::Device { device, ptr, size } => regions.push(Region {
                    device: *device,
                    ptr: *ptr,
                    size: *size,
                    slot: None,
                }),
                Shard::Host { .. } => panic!(
                    "mod2: host placement {op:?} reached a device kernel (feed device input regions first)"
                ),
            }
        }
        regions
    }

    /// Compile one batch into a replayable [`ExecutionGraph`]: sweep
    /// completed launches, assign interior regions (reusing dead ranges with
    /// reuse edges), emit def-use plus boundary-alias edges, build the
    /// single-device partition, hand it to executable build, and record
    /// batch-terminal defs as exports (regions assigned at first replay).
    /// Interior regions pin for executable lifetime (one-shot graphs carry
    /// them into their completion record at replay; persistent graphs until
    /// [`ExecutionGraph::dispose`]). Same shapes required across replays;
    /// shape change means recompile.
    pub fn compile(
        &self,
        batch: Vec<Cmd>,
        boundary: Map<OpId, Arc<Placement>>,
        persistent: bool,
    ) -> ExecutionGraph {
        let mut guard = Self::arenas();
        let arenas: &mut Arenas = &mut guard;
        Self::sweep(arenas);

        let mut def_idx: Map<OpId, usize> = Map::default();
        for (i, cmd) in batch.iter().enumerate() {
            for op in cmd.defs() {
                if def_idx.insert(op, i).is_some() {
                    assert!(
                        boundary.contains_key(&op),
                        "mod2: duplicate def of interior op {op:?} (not SSA)"
                    );
                }
            }
        }
        let mut last_use: Map<OpId, usize> = Map::default();
        for (i, cmd) in batch.iter().enumerate() {
            for op in cmd.reads() {
                last_use.insert(op, i);
            }
        }
        let last_touch = |op: OpId, last_use: &Map<OpId, usize>, def_idx: &Map<OpId, usize>| -> usize {
            last_use.get(&op).copied().or_else(|| def_idx.get(&op).copied()).expect("mod2: op without def")
        };

        let mut resolved: Map<OpId, Vec<Region>> = Map::default();
        let mut writer: Map<OpId, usize> = Map::default();
        let mut alias: Map<*const Placement, usize> = Map::default();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let edge = |from: usize, to: usize, edges: &mut Vec<(usize, usize)>| {
            if from != to && !edges.contains(&(from, to)) {
                edges.push((from, to));
            }
        };
        let touch_alias = |placement: &Arc<Placement>, i: usize, alias: &mut Map<*const Placement, usize>, edges: &mut Vec<(usize, usize)>| {
            let id = Arc::as_ptr(placement);
            if let Some(prev) = alias.get(&id) {
                if *prev != i && !edges.contains(&(*prev, i)) {
                    edges.push((*prev, i));
                }
            }
            alias.insert(id, i);
        };
        let resolve = |resolved: &Map<OpId, Vec<Region>>, boundary: &Map<OpId, Arc<Placement>>, op: OpId| -> (Vec<Region>, Option<OpId>) {
            if let Some(regions) = resolved.get(&op) {
                return (regions.clone(), None);
            }
            if let Some(placement) = boundary.get(&op) {
                return (Self::regions_of(placement, op), Some(op));
            }
            panic!("mod2: use of unknown op {op:?} (never defined or missing binding)");
        };

        let mut nodes: Vec<PartitionOp> = Vec::new();
        let mut boundary_set: Map<OpId, ()> = Map::default();
        let mut exports: Vec<OpId> = Vec::new();
        let mut output_specs: Vec<OutputSpec> = Vec::new();
        let mut claimed: Vec<(Dev, u32, u64, u64)> = Vec::new();
        let mut exported_slots: Vec<(Dev, u32, u64)> = Vec::new();
        let mut pool: Vec<(Region, Option<OpId>)> = Vec::new();

        for (i, cmd) in batch.iter().enumerate() {
            match cmd {
                Cmd::Launch { program, args, outputs } => {
                    let kinds = Self::program_params(*program);
                    assert!(args.len() <= kinds.len(), "mod2: more launch args than program params");
                    let device = program.dev;
                    let mut node_args = Vec::with_capacity(args.len());
                    for (j, op) in args.iter().enumerate() {
                        match kinds[j] {
                            ParamKind::Global | ParamKind::GlobalMut => {
                                let (regions, is_boundary) = resolve(&resolved, &boundary, *op);
                                if let Some(global) = is_boundary {
                                    let placement = boundary
                                        .get(&global)
                                        .expect("mod2: boundary left the table mid-compile");
                                    touch_alias(placement, i, &mut alias, &mut edges);
                                    boundary_set.insert(global, ());
                                    node_args.push(ArgRef::Rebind(global));
                                } else {
                                    node_args.push(ArgRef::Fixed(regions));
                                }
                                if let Some(w) = writer.get(op) {
                                    edge(*w, i, &mut edges);
                                }
                            }
                            ParamKind::Variable => node_args.push(ArgRef::Scalar(Self::scalar_value(*op))),
                        }
                    }
                    let mut node_outputs = Vec::with_capacity(outputs.len());
                    for (k, op) in outputs.iter().enumerate() {
                        let size = Self::output_bytes(*program, k);
                        if boundary.contains_key(op) {
                            let placement = boundary
                                .get(op)
                                .expect("mod2: mutation target left the boundary table");
                            let regions = Self::regions_of(placement, *op);
                            let total: u64 = regions.iter().map(|r| r.size).sum();
                            assert!(total == size, "mod2: in-place mutation size mismatch on {op:?}");
                            touch_alias(placement, i, &mut alias, &mut edges);
                            boundary_set.insert(*op, ());
                            node_outputs.push(ArgRef::Rebind(*op));
                        } else {
                            let (region, source) =
                                Self::assign(arenas, &mut pool, device, size);
                            if let Some(dead) = source {
                                edge(last_touch(dead, &last_use, &def_idx), i, &mut edges);
                            }
                            if let Some(slot) = region.slot {
                                claimed.push((device, slot.0, slot.1, region.size));
                            }
                            node_outputs.push(ArgRef::Rebind(*op));
                            resolved.insert(*op, vec![region]);
                            if last_use.get(op).is_none() {
                                if let Some(slot) = region.slot {
                                    exported_slots.push((device, slot.0, slot.1));
                                }
                                exports.push(*op);
                                output_specs.push(OutputSpec { op: *op, size, device });
                            }
                        }
                        writer.insert(*op, i);
                    }
                    nodes.push(PartitionOp { program: Some(*program), device, args: node_args, outputs: node_outputs });
                    Self::drop_dead(&mut resolved, &mut pool, &last_use, &boundary, i);
                }
                Cmd::Copy { src, dst } => {
                    let (src_regions, src_boundary) = resolve(&resolved, &boundary, *src);
                    let device = src_regions.first().expect("mod2: copy of a sizeless value").device;
                    assert!(
                        src_regions.iter().all(|r| r.device == device),
                        "mod2: multi-device copy source (staging todo)"
                    );
                    let src_ref = if let Some(global) = src_boundary {
                        let placement = boundary
                            .get(&global)
                            .expect("mod2: boundary left the table mid-compile");
                        touch_alias(placement, i, &mut alias, &mut edges);
                        boundary_set.insert(global, ());
                        ArgRef::Rebind(global)
                    } else {
                        ArgRef::Fixed(src_regions.clone())
                    };
                    if let Some(w) = writer.get(src) {
                        edge(*w, i, &mut edges);
                    }
                    let size: u64 = src_regions.iter().map(|r| r.size).sum();
                    let dst_ref = if boundary.contains_key(dst) {
                        let placement =
                            boundary.get(dst).expect("mod2: copy target left the boundary table");
                        let regions = Self::regions_of(placement, *dst);
                        let total: u64 = regions.iter().map(|r| r.size).sum();
                        assert!(total == size, "mod2: in-place copy size mismatch on {dst:?}");
                        touch_alias(placement, i, &mut alias, &mut edges);
                        boundary_set.insert(*dst, ());
                        ArgRef::Rebind(*dst)
                    } else {
                        let (region, source) = Self::assign(arenas, &mut pool, device, size);
                        if let Some(dead) = source {
                            edge(last_touch(dead, &last_use, &def_idx), i, &mut edges);
                        }
                        if let Some(slot) = region.slot {
                            claimed.push((device, slot.0, slot.1, region.size));
                        }
                        resolved.insert(*dst, vec![region]);
                        if last_use.get(dst).is_none() {
                            if let Some(slot) = region.slot {
                                exported_slots.push((device, slot.0, slot.1));
                            }
                            exports.push(*dst);
                            output_specs.push(OutputSpec { op: *dst, size, device });
                        }
                        ArgRef::Rebind(*dst)
                    };
                    writer.insert(*dst, i);
                    nodes.push(PartitionOp { program: None, device, args: vec![src_ref], outputs: vec![dst_ref] });
                    Self::drop_dead(&mut resolved, &mut pool, &last_use, &boundary, i);
                }
            }
        }

        let device = nodes.first().expect("mod2: empty batch").device;
        assert!(
            nodes.iter().all(|n| n.device == device),
            "mod2: multi-device batch (cross-device event sync todo)"
        );
        let mut boundary_ids: Vec<OpId> = boundary_set.into_keys().collect();
        boundary_ids.sort();
        let partition = Partition { device, ops: nodes, edges };
        let exec = Self::build_exec(partition);
        let interior: Vec<(Dev, u32, u64, u64)> = claimed
            .into_iter()
            .filter(|c| !exported_slots.contains(&(c.0, c.1, c.2)))
            .collect();

        ExecutionGraph {
            execs: vec![exec],
            devices: vec![device],
            boundary: boundary_ids,
            outputs: output_specs,
            exports,
            persistent,
            interior,
            prev_outputs: RefCell::new(Vec::new()),
        }
    }

    /// Assign one fresh interior region: graph-local pool first, else global
    /// free lists, else bump, else grow. Returns the region plus the dead op
    /// it was reused from, if any.
    fn assign(
        arenas: &mut Arenas,
        pool: &mut Vec<(Region, Option<OpId>)>,
        device: Dev,
        size: u64,
    ) -> (Region, Option<OpId>) {
        let size = align_up(size, ARENA_ALIGN);
        let mut best: Option<usize> = None;
        for (i, (region, _)) in pool.iter().enumerate() {
            if region.device == device && region.size >= size {
                if best.is_none_or(|b| pool[b].0.size > region.size) {
                    best = Some(i);
                }
            }
        }
        if let Some(i) = best {
            return pool.swap_remove(i);
        }
        loop {
            if arenas.vec.iter().all(|a| a.device != device) {
                arenas.vec.push(Arena::new(device));
            }
            let arena =
                arenas.vec.iter_mut().find(|a| a.device == device).expect("mod2: arena just pushed");
            if let Some((chunk, offset, ptr, source)) = arena.alloc(size) {
                let size = align_up(size, ARENA_ALIGN);
                return (Region { device, ptr, size, slot: Some((chunk, offset)) }, source);
            }
            Self::grow(device, size);
        }
    }

    /// Drop fresh interior values whose last use was command `i` into the
    /// graph-local pool (reused within this compile). Boundary values never
    /// leave the caller table here.
    fn drop_dead(
        resolved: &mut Map<OpId, Vec<Region>>,
        pool: &mut Vec<(Region, Option<OpId>)>,
        last_use: &Map<OpId, usize>,
        boundary: &Map<OpId, Arc<Placement>>,
        i: usize,
    ) {
        let dead: Vec<OpId> = resolved
            .keys()
            .filter(|op| last_use.get(*op).is_none_or(|u| *u == i) && !boundary.contains_key(*op))
            .copied()
            .collect();
        for op in dead {
            let regions = resolved.remove(&op).expect("mod2: resolved entry vanished");
            for region in regions {
                if region.slot.is_some() {
                    pool.push((region, Some(op)));
                }
            }
        }
    }
}

/// Compiled, replayable work: one backend-native executable per device
/// partition, replayed serially. Interior addresses frozen at build;
/// boundary and output addresses bind per replay from fresh placements.
/// Same shapes required across replays; shape change means recompile.
#[derive(Debug)]
pub struct ExecutionGraph {
    execs: Vec<BackendExec>,
    devices: Vec<Dev>,
    /// Boundary ops referenced by any node: replay bindings required.
    boundary: Vec<OpId>,
    outputs: Vec<OutputSpec>,
    /// Batch-terminal defs, in def order. The caller records these to
    /// reference the values from later batches.
    exports: Vec<OpId>,
    /// One-shot graphs are consumed by [`Self::replay_once`]; persistent
    /// graphs replay repeatedly and release this at [`Self::dispose`].
    persistent: bool,
    /// Pinned interior regions: every region the executable can touch.
    /// One-shot graphs move these into their completion record at replay;
    /// persistent graphs hold them until dispose.
    interior: Vec<(Dev, u32, u64, u64)>,
    /// Previous replay's output placements, for liveness-gated slot reuse:
    /// a slot is reused iff its `Arc` is held only here.
    prev_outputs: RefCell<Vec<(OpId, Arc<Placement>)>>,
}

impl ExecutionGraph {
    /// Batch-terminal defs, in def order.
    pub fn exports(&self) -> &[OpId] {
        &self.exports
    }

    /// Shared replay core: check binding coverage, assign output regions
    /// (reusing dead previous slots), and build the combined bindings map.
    /// Returns outputs plus bindings; the caller launches and records.
    fn bind(
        &self,
        arenas: &mut Arenas,
        inputs: &Map<OpId, Arc<Placement>>,
    ) -> (Map<OpId, Arc<Placement>>, Map<OpId, Vec<Region>>) {
        assert!(
            inputs.len() == self.boundary.len() && self.boundary.iter().all(|op| inputs.contains_key(op)),
            "mod2: replay bindings must cover exactly the graph boundary"
        );
        let mut bindings: Map<OpId, Vec<Region>> = Map::default();
        for op in &self.boundary {
            let placement = inputs.get(op).expect("mod2: boundary op missing from replay bindings");
            bindings.insert(*op, Scheduler::regions_of(placement, *op));
        }
        let mut outputs: Map<OpId, Arc<Placement>> = Map::default();
        let mut prev = self.prev_outputs.borrow_mut();
        for spec in &self.outputs {
            let regions = match prev.iter().find(|(op, _)| *op == spec.op) {
                Some((_, placement)) if Arc::strong_count(placement) == 1 => {
                    Scheduler::regions_of(placement, spec.op)
                }
                _ => {
                    let (region, _) = Self::assign_output(arenas, spec);
                    vec![region]
                }
            };
            let shards = regions
                .iter()
                .map(|region| Shard::Device { device: region.device, ptr: region.ptr, size: region.size })
                .collect();
            let placement = Arc::new(Placement { shards });
            if let Some(slot) = prev.iter_mut().find(|(op, _)| *op == spec.op) {
                let old = std::mem::replace(&mut slot.1, placement.clone());
                Self::retire(arenas, old);
            } else {
                prev.push((spec.op, placement.clone()));
            }
            bindings.insert(spec.op, Scheduler::regions_of(&placement, spec.op));
            outputs.insert(spec.op, placement);
        }
        drop(prev);
        (outputs, bindings)
    }

    /// Assign one output region: free lists, bump, grow.
    fn assign_output(arenas: &mut Arenas, spec: &OutputSpec) -> (Region, Option<OpId>) {
        loop {
            if arenas.vec.iter().all(|a| a.device != spec.device) {
                arenas.vec.push(Arena::new(spec.device));
            }
            let arena =
                arenas.vec.iter_mut().find(|a| a.device == spec.device).expect("mod2: arena just pushed");
            if let Some((chunk, offset, ptr, source)) = arena.alloc(spec.size) {
                let size = align_up(spec.size, ARENA_ALIGN);
                return (Region { device: spec.device, ptr, size, slot: Some((chunk, offset)) }, source);
            }
            Scheduler::grow(spec.device, spec.size);
        }
    }

    /// Retire a replaced previous output: freed behind the device's last
    /// launch event if it is otherwise dead (immediately if nothing ever
    /// launched — nothing is in flight then). A still-referenced placement
    /// is left alone; its owner frees it.
    fn retire(arenas: &mut Arenas, old: Arc<Placement>) {
        if Arc::strong_count(&old) != 1 {
            return;
        }
        for shard in &old.shards {
            if let Shard::Device { device, ptr, size } = shard {
                let slot = Scheduler::find_region(arenas, *device, *ptr, *size);
                match Scheduler::last_event(arenas, *device) {
                    None => Scheduler::arena_for(arenas, *device).release(slot.0, slot.1, *size, None),
                    Some(event) => arenas.pending.push(PendingLaunch {
                        device: *device,
                        event,
                        regions: vec![(slot.0, slot.1, *size)],
                        exec: None,
                        prev: vec![old.clone()],
                    }),
                }
            }
        }
    }

    /// Replay a persistent graph: bind, launch every executable
    /// asynchronously, record completion events. The executables stay;
    /// interior stays pinned; previous outputs gate slot reuse.
    pub fn replay(&self, inputs: &Map<OpId, Arc<Placement>>) -> Map<OpId, Arc<Placement>> {
        assert!(self.persistent, "mod2: replay is for persistent graphs (one-shot graphs use replay_once)");
        let mut guard = Scheduler::arenas();
        let arenas: &mut Arenas = &mut guard;
        let (outputs, bindings) = self.bind(arenas, inputs);
        let mut events = Vec::with_capacity(self.execs.len());
        for exec in &self.execs {
            events.push(Scheduler::launch_exec(exec, &bindings));
        }
        for (i, event) in events.into_iter().enumerate() {
            let device = self.devices.get(i).copied().expect("mod2: exec without device");
            Self::record_event(arenas, device, event);
        }
        drop(guard);
        outputs
    }

    /// Replay a one-shot graph: bind, launch, move interior regions plus
    /// executables into the completion record (freed/destroyed by the sweep
    /// once the launch finishes), and return fresh outputs. The graph is
    /// consumed; it cannot replay or dispose afterwards.
    pub fn replay_once(self, inputs: &Map<OpId, Arc<Placement>>) -> Map<OpId, Arc<Placement>> {
        assert!(!self.persistent, "mod2: replay_once is for one-shot graphs (persistent graphs use replay)");
        let mut guard = Scheduler::arenas();
        let arenas: &mut Arenas = &mut guard;
        let (outputs, bindings) = self.bind(arenas, inputs);
        let mut events = Vec::with_capacity(self.execs.len());
        for exec in &self.execs {
            events.push(Scheduler::launch_exec(exec, &bindings));
        }
        let interior = self.interior;
        let mut placed: Vec<(u32, u64, u64)> = Vec::new();
        for (i, exec) in self.execs.into_iter().enumerate() {
            let device = self.devices[i];
            let event = events[i];
            Self::record_event(arenas, device, event);
            let mut regions = Vec::new();
            for (d, c, o, s) in &interior {
                if *d == device && !placed.contains(&(*c, *o, *s)) {
                    placed.push((*c, *o, *s));
                    regions.push((*c, *o, *s));
                }
            }
            arenas.pending.push(PendingLaunch { device, event, regions, exec: Some(exec), prev: Vec::new() });
        }
        drop(guard);
        outputs
    }

    /// Record a launch completion event as the device's latest.
    fn record_event(arenas: &mut Arenas, device: Dev, event: u64) {
        if let Some(slot) = arenas.last_event.iter_mut().find(|(d, _)| *d == device) {
            slot.1 = event;
        } else {
            arenas.last_event.push((device, event));
        }
    }

    /// Dispose a persistent graph: synchronize its devices (the one allowed
    /// sync, off the hot path), sweep everything completed, return pinned
    /// interior regions plus dead previous outputs to the free lists. After
    /// this the executables must never run again. One-shot graphs never
    /// reach here (consumed by `replay_once`).
    pub fn dispose(self) {
        assert!(self.persistent, "mod2: one-shot graphs are consumed by replay_once, not disposed");
        for device in &self.devices {
            Scheduler::sync_device(*device);
        }
        let mut guard = Scheduler::arenas();
        let arenas: &mut Arenas = &mut guard;
        Scheduler::sweep(arenas);
        for (device, chunk, offset, size) in self.interior {
            Scheduler::arena_for(arenas, device).release(chunk, offset, size, None);
        }
        for (_, placement) in self.prev_outputs.into_inner() {
            if Arc::strong_count(&placement) == 1 {
                for shard in &placement.shards {
                    if let Shard::Device { device, ptr, size } = shard {
                        let (chunk, offset) = Scheduler::find_region(arenas, *device, *ptr, *size);
                        Scheduler::arena_for(arenas, *device).release(chunk, offset, *size, None);
                    }
                }
            }
        }
    }
}
