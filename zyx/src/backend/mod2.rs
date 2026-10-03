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

use super::{Dev, ProgramId};
use crate::{
    Map,
    kernel::{OpId, ParamKind},
};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// One device-resident (or host-resident) piece of a placed value.
#[derive(Debug, Clone)]
pub enum Shard {
    /// Host-owned bytes (loaded data, weights staging, readback targets).
    Host { data: Vec<u8> },
    /// A region inside a scheduler arena on `device`.
    Device { device: Dev, ptr: u64, size: u64 },
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
///
/// `Launch` carries its program metadata inline (the scheduler holds no
/// kernels): `params` are the per-`Param` kinds in head order, `out_bytes`
/// the per-output byte sizes, `scalars` the variable op values.
#[derive(Debug)]
pub enum Cmd {
    /// Run `program`. Devices come from `program.dev`; arg regions must
    /// agree (cross-device staging of one value into one node is later work).
    Launch {
        program: ProgramId,
        args: Vec<OpId>,
        outputs: Vec<OpId>,
        params: Vec<ParamKind>,
        out_bytes: Vec<u64>,
        scalars: Vec<(OpId, i64)>,
    },
    /// Copy from `src` value to `dst` value. `device` names the destination
    /// device: same-device fresh `dst`s are assigned there; a cross-device
    /// fresh `dst` is rejected (cross into a caller pre-placed boundary
    /// value). Cross-device copies stage through host temps or go peer
    /// (same vendor). Host-resident sources never become nodes: they upload
    /// eagerly around capture/launch.
    Copy { src: OpId, dst: OpId, device: Dev },
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

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.cmds.len()
    }
}

impl CmdQueue {
    fn schedule(self) -> Plan {
        todo!()
    }
}

struct Plan {
    partitions: Vec<PlanPartition>,
}

enum PlanPartition {
    CudaGraph,
    VulkanPipeline,
    CpuGraph,
    // and so on
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

enum Buffer {}
