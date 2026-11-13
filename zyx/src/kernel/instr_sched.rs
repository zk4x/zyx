// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Instruction scheduling for kernel optimization.
//!
//! This module provides instruction scheduling optimizations for kernels,
//! including:
//!
//! - Ordering params and storages (global read-only, global read-write, local read-only,
//!   local read-write) at the beginning of the kernel.
//! - Topologically sorting the remaining operations by dependency so that
//!   operations which depend on the fewest other operations come first.
//! - Improving instruction pipeline utilization.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::{
    Map,
    kernel::{Kernel, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind, TTOp},
};

impl Kernel {
    /// Schedule instructions for better instruction throughput.
    ///
    /// This method reorders kernel operations to improve instruction
    /// scheduling. The final order is:
    ///
    /// 1. All global read-only params, preserving their order.
    /// 2. All GlobalMut params, preserving their order.
    /// 3. All local read-only storages, preserving their order.
    /// 4. All local read-write storages, preserving their order.
    /// 5. The remaining operations, topologically sorted by dependency with
    ///    operations that depend on the fewest other operations first.
    ///
    /// Memory operations that share a param or storage keep their relative order, stores
    /// are never moved out of the loops or if blocks that contain them, and
    /// stores and loads are never moved before barriers.
    pub fn instruction_schedule(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("instruction_schedule");
        let mut global_ro = Vec::new();
        let mut global_rw = Vec::new();
        let mut local_rw = Vec::new();
        let mut rest = Vec::new();

        // A tiled device keeps its params' shapes, and a shape is an operand
        // of its param: its cone must stay before the params, even though the
        // schedule groups the params first. Empty on every other device.
        let mut shape_meta: Vec<OpId> = Vec::new();
        let mut shape_cone: crate::Set<OpId> = crate::Set::default();
        for (_, node) in self.iter_unordered() {
            if let Op::Param { shape, .. } = node {
                let mut stack = vec![*shape];
                while let Some(id) = stack.pop() {
                    if id.is_null() || !shape_cone.insert(id) {
                        continue;
                    }
                    stack.extend(self.ops[id].op.parameters());
                }
            }
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if shape_cone.contains(&op_id) {
                shape_meta.push(op_id);
                op_id = next;
                continue;
            }
            match self.at(op_id) {
                Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } => global_ro.push(op_id),
                Op::Param { kind: ParamKind::GlobalMut, .. } => global_rw.push(op_id),
                Op::Storage { scope: MemScope::Local, .. } => local_rw.push(op_id),
                _ => rest.push(op_id),
            }
            op_id = next;
        }

        let sorted_rest = self.schedule_rest(&rest);

        let mut order =
            Vec::with_capacity(shape_meta.len() + global_ro.len() + global_rw.len() + local_rw.len() + sorted_rest.len());
        order.extend(shape_meta);
        order.extend(global_ro);
        order.extend(global_rw);
        order.extend(local_rw);
        order.extend(sorted_rest);

        if order.is_empty() {
            return;
        }
        for window in order.windows(2) {
            self.ops[window[0]].next = window[1];
            self.ops[window[1]].prev = window[0];
        }
        let first = order[0];
        let last = order[order.len() - 1];
        self.ops[first].prev = OpId::NULL;
        self.ops[last].next = OpId::NULL;
        self.head = first;
        self.tail = last;

        #[cfg(debug_assertions)]
        self.verify();
    }

    /// Topologically sort the non-memory operations by dependency.
    ///
    /// The sort respects the following precedence constraints:
    ///
    /// - A use must come after its declaration.
    /// - Loads and stores that share a param or storage keep their relative order, so
    ///   memory hazards (RAW, WAR, WAW) on the same buffer are preserved.
    /// - Stores never leave the loops or if blocks that contain them and never
    ///   cross barriers.
    /// - Loads never cross barriers.
    /// - Control flow and barriers keep their mutual order.
    ///
    /// Everything else is free to move; among the ready operations those that
    /// depend on the fewest other operations are emitted first, with ties
    /// broken by original position.
    fn schedule_rest(&self, rest: &[OpId]) -> Vec<OpId> {
        let n = rest.len();
        if n == 0 {
            return Vec::new();
        }

        // OpId → position in rest. OpId is u32 so we can use a Vec. Size to the
        // full slab range so rest ops can reference params/storages (or ops removed to
        // the front) without an out-of-bounds access.
        let max_id = self.ops.max_id().0 as usize;
        let mut idx = vec![usize::MAX; max_id + 1];
        for (i, &id) in rest.iter().enumerate() {
            idx[id.0 as usize] = i;
        }

        let mut structural = vec![false; n];
        let mut barrier = vec![false; n];
        let mut store = vec![false; n];
        let mut load = vec![false; n];
        // The latency each op's result takes to become usable, used to rank
        // ready ops by critical path. A global load pays the memory round trip
        // (several hundred cycles), a shared fragment load the shared latency,
        // a tensor-core instruction a few cycles, and everything else is
        // integer/ALU work of a couple of cycles.
        let mut latency = vec![4u64; n];
        // Shared-memory fragment loads whose results occupy registers: only a
        // few may be in flight at once (see `FRAGMENT_SLOTS`).
        let mut frag_load = vec![false; n];
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Barrier
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::DmRead(_))
                | Op::TT(TTOp::DmWrite(_))
                | Op::TT(TTOp::DmReadTransposed(_))
                | Op::TT(TTOp::DmReadBroadcastCol(_))
                | Op::TT(TTOp::DmFill(_))
                | Op::TT(TTOp::DmCopyWords(_))
                | Op::TT(TTOp::DmIndexPick(_))
                | Op::TT(TTOp::DmKernel(_))
                | Op::TT(TTOp::DmWait)
                | Op::TT(TTOp::TensixGrid(_))
                | Op::TT(TTOp::Unpack(_))
                | Op::TT(TTOp::Math(_))
                | Op::TT(TTOp::Pack(_))
                | Op::TT(TTOp::Reader(_))
                | Op::TT(TTOp::Writer(_)) => {
                    structural[i] = true;
                    barrier[i] = true;
                    latency[i] = 1;
                }
                Op::Loop { .. } | Op::EndLoop => {
                    structural[i] = true;
                    latency[i] = 1;
                }
                // Opaque side effect (inline asm, TT LLK call):
                // program-ordered like a barrier for placement (never
                // crosses one), but not a barrier itself.
                Op::Asm { .. } => {
                    structural[i] = true;
                    latency[i] = 1;
                }
                Op::TT(TTOp::Sfpu { .. })
                | Op::TT(TTOp::SfpLoad(_))
                | Op::TT(TTOp::SfpStore(_))
                | Op::TT(TTOp::SfpConst(_))
                | Op::TT(TTOp::Sfp(_))
                | Op::TT(TTOp::Replay { .. })
                | Op::TT(TTOp::Unpacr { .. })
                | Op::TT(TTOp::Setadczw { .. })
                | Op::TT(TTOp::Setadcx { .. })
                | Op::TT(TTOp::Setrwc { .. })
                | Op::TT(TTOp::Zeroacc { .. })
                | Op::TT(TTOp::Mvmul { .. })
                | Op::TT(TTOp::Setc16 { .. })
                | Op::TT(TTOp::Post { .. })
                | Op::TT(TTOp::Wait { .. })
                | Op::TT(TTOp::WaitBelowMax { .. }) => {
                    structural[i] = true;
                    latency[i] = 1;
                }
                Op::Store { .. } => {
                    store[i] = true;
                    latency[i] = 8;
                }
                Op::Load { src, .. } => {
                    load[i] = true;
                    // A parameter is memory; a storage is a register move.
                    latency[i] = if matches!(self.at(*src), Op::Param { .. }) { 400 } else { 4 };
                }
                Op::Wmma { .. } => latency[i] = 12,
                // `ldmatrix` reads local storage the loop refills: pin it
                // inside the loop exactly like a load of that storage. Its
                // result is a fragment, so it takes a fragment slot.
                Op::PTX(p) => match p.as_ref() {
                    // Opaque pre-assembled PTX: ordered like inline asm.
                    PTXOp::Bytes(_) => {
                        structural[i] = true;
                        latency[i] = 1;
                    }
                    PTXOp::VLdMatrix { .. } | PTXOp::LdMatrix { .. } => {
                        load[i] = true;
                        frag_load[i] = true;
                        latency[i] = 30;
                    }
                    // A warp barrier orders the shared accesses around it.
                    PTXOp::WarpSync => {
                        structural[i] = true;
                        barrier[i] = true;
                        latency[i] = 1;
                    }
                    // A shuffle reads two registers and writes one: no shared
                    // state to order against.
                    PTXOp::Shfl { .. } => latency[i] = 20,
                    // The shared window base and its generic-address form: one
                    // definition each, carrying no state of their own.
                    PTXOp::CvtaShared { .. } | PTXOp::SharedBase { .. } => latency[i] = 1,
                    // A shared load feeds registers from the tile the loop
                    // refills, so it pins like an `ldmatrix`.
                    PTXOp::LdShared { .. } => {
                        load[i] = true;
                        latency[i] = 20;
                    }
                    // The fill's own store into that tile.
                    PTXOp::StShared { .. } => {
                        store[i] = true;
                        latency[i] = 8;
                    }
                },
                // A copy reads its src and writes its dst: it pins like a
                // load (barriers, asm, loops) and orders like a store.
                Op::Copy { .. } | Op::TT(TTOp::Copy { .. }) => {
                    store[i] = true;
                    load[i] = true;
                }
                _ => {}
            }
        }

        // Precedence edges are collected as `(u, v)` pairs meaning `u` must be
        // ordered before `v`, then laid out in CSR form to avoid one heap
        // allocation per operation.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut in_degree = vec![0usize; n];
        // The operands of every op in `rest`, flattened as `rest` indices: the
        // data edges and, for the priority below, the record of when a
        // producer is consumed.
        let mut params_flat: Vec<usize> = Vec::new();
        let mut param_start = vec![0usize; n + 1];

        // Uses must come after their declarations. Iterate params directly
        // without collecting into a Vec.
        for (i, &id) in rest.iter().enumerate() {
            macro_rules! add_param {
                ($p:expr) => {{
                    if !$p.is_null() {
                        let j = idx[$p.0 as usize];
                        if j != usize::MAX {
                            params_flat.push(j);
                            edges.push((j, i));
                            in_degree[i] += 1;
                        }
                    }
                }};
            }
            match self.at(id) {
                Op::Cast { x, .. }
                | Op::Bitcast { x, .. }
                | Op::Unary { x, .. }
                | Op::Reshape { x, .. }
                | Op::Pad { x, .. }
                | Op::Permute { x, .. }
                | Op::Expand { x, .. }
                | Op::Flip { x, .. }
                | Op::Narrow { x, .. }
                | Op::Reduce { x, .. } => {
                    add_param!(x)
                }
                Op::Binary { x, y, .. } => {
                    add_param!(x);
                    add_param!(y);
                }
                Op::Select { cond, if_true, if_false } => {
                    add_param!(cond);
                    add_param!(if_true);
                    add_param!(if_false);
                }
                Op::Param { .. }
                | Op::Const(_)
                | Op::Unroll { .. }
                | Op::Storage { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::DmRead(_))
                | Op::TT(TTOp::DmWrite(_))
                | Op::TT(TTOp::DmReadTransposed(_))
                | Op::TT(TTOp::DmReadBroadcastCol(_))
                | Op::TT(TTOp::DmFill(_))
                | Op::TT(TTOp::DmCopyWords(_))
                | Op::TT(TTOp::DmIndexPick(_))
                | Op::TT(TTOp::DmKernel(_))
                | Op::TT(TTOp::DmWait)
                | Op::TT(TTOp::TensixGrid(_))
                | Op::TT(TTOp::Setadczw { .. })
                | Op::TT(TTOp::Setadcx { .. })
                | Op::TT(TTOp::Zeroacc { .. })
                | Op::TT(TTOp::Mvmul { .. })
                | Op::TT(TTOp::Setc16 { .. })
                | Op::TT(TTOp::Setcfg { .. })
                | Op::TT(TTOp::Post { .. })
                | Op::TT(TTOp::Wait { .. })
                | Op::TT(TTOp::WaitBelowMax { .. })
                | Op::TT(TTOp::Semset { .. })
                | Op::TT(TTOp::Replay { .. })
                | Op::TT(TTOp::Unpack(_))
                | Op::TT(TTOp::Math(_))
                | Op::TT(TTOp::Pack(_))
                | Op::TT(TTOp::Reader(_))
                | Op::TT(TTOp::Writer(_))
                | Op::TT(TTOp::ReserveBack { .. })
                | Op::TT(TTOp::PushBack { .. })
                | Op::TT(TTOp::WaitFront { .. })
                | Op::TT(TTOp::PopFront { .. }) => {}
                Op::Source(_)
                | Op::GPU(_)
                | Op::Spirv(_)
                | Op::After { .. }
                | Op::ToDevice { .. }
                | Op::Contiguous { .. }
                | Op::Program { .. }
                | Op::Kernel(_) => {
                    todo!()
                }
                // PTX-level instruction: schedule it after the storage and
                // index it reads.
                Op::PTX(p) => match p.as_ref() {
                    // Pre-assembled PTX is opaque: it pins nowhere and carries
                    // no operand the scheduler can name.
                    PTXOp::Bytes(_) => {}
                    PTXOp::VLdMatrix { src, addr, .. } => {
                        add_param!(src);
                        add_param!(addr);
                    }
                    // No operands: ordered by the barrier edges alone.
                    PTXOp::WarpSync => {}
                    PTXOp::Shfl { value, lane, .. } => {
                        add_param!(value);
                        add_param!(lane);
                    }
                    PTXOp::LdMatrix { addr, .. } => add_param!(addr),
                    PTXOp::CvtaShared { addr } => add_param!(addr),
                    PTXOp::SharedBase { buf } => add_param!(buf),
                    PTXOp::LdShared { buf, addr } => {
                        add_param!(buf);
                        add_param!(addr);
                    }
                    PTXOp::StShared { buf, addr, src } => {
                        add_param!(buf);
                        add_param!(addr);
                        add_param!(src);
                    }
                },
                // Graph-only marker, erased at kernelization: never present.
                Op::Detach { .. } => {
                    todo!()
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => add_param!(len),
                    RangeKind::Local(_) => {}
                },
                Op::Store { dst, dst_idx, src: x } => {
                    add_param!(dst);
                    add_param!(dst_idx);
                    add_param!(x);
                }
                Op::Copy { src, src_idx, dst, dst_idx } => {
                    add_param!(src);
                    add_param!(src_idx);
                    add_param!(dst);
                    add_param!(dst_idx);
                }
                Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, .. }) => {
                    add_param!(src);
                    add_param!(src_idx);
                    add_param!(dst);
                    add_param!(dst_idx);
                }
                Op::Load { src, src_idx } => {
                    add_param!(src);
                    add_param!(src_idx);
                }
                Op::Loop { len, .. } => add_param!(len),
                Op::Wmma { a, b, c, .. } => {
                    add_param!(a);
                    add_param!(b);
                    add_param!(c);
                }
                Op::Asm { ops, .. } => {
                    for &p in ops.iter() {
                        add_param!(p);
                    }
                }
                Op::TT(TTOp::Unpacr { src, .. }) => {
                    add_param!(*src);
                }
                Op::TT(TTOp::Setrwc { dst }) => {
                    add_param!(*dst);
                }
                Op::TT(TTOp::Sfpu { a, b, slot, .. }) => {
                    add_param!(a);
                    if !b.is_null() {
                        add_param!(b);
                    }
                    add_param!(slot);
                }
                Op::TT(TTOp::SfpLoad(load)) => {
                    add_param!(load.slot);
                }
                Op::TT(TTOp::SfpStore(store)) => {
                    add_param!(store.slot);
                }
                Op::TT(TTOp::SfpConst(_)) | Op::TT(TTOp::Sfp(_)) => {}
                Op::Stack { ops } => {
                    for &p in ops.iter() {
                        add_param!(p);
                    }
                }
                Op::Index { vec, .. } => add_param!(vec),
            }
            param_start[i + 1] = params_flat.len();
        }

        // Loads, stores, and copies to the same buffer keep their relative
        // order. Locations name their buffer directly now: group by the
        // addressed buffer, or same-buffer traffic through different
        // accesses could reorder across FIFO accounting.
        let mut by_memory_target: Map<OpId, Vec<usize>> = Map::default();
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Load { src, .. } => {
                    by_memory_target.entry(*src).or_default().push(i);
                }
                Op::Store { dst, .. } => {
                    by_memory_target.entry(*dst).or_default().push(i);
                }
                Op::Copy { src, dst, .. } => {
                    by_memory_target.entry(*src).or_default().push(i);
                    by_memory_target.entry(*dst).or_default().push(i);
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    by_memory_target.entry(*src).or_default().push(i);
                    by_memory_target.entry(*dst).or_default().push(i);
                }
                Op::TT(TTOp::Unpacr { src, .. }) => {
                    by_memory_target.entry(*src).or_default().push(i);
                }
                Op::TT(TTOp::Setrwc { dst }) => {
                    by_memory_target.entry(*dst).or_default().push(i);
                }
                Op::TT(TTOp::Sfpu { a, b, slot, .. }) => {
                    for f in [a, b, slot] {
                        if !f.is_null() {
                            by_memory_target.entry(*f).or_default().push(i);
                        }
                    }
                }
                // The lowered shared traffic names its storage directly, so it
                // keeps its relative order with every other access to it.
                Op::PTX(p) => match p.as_ref() {
                    PTXOp::LdShared { buf, .. } | PTXOp::StShared { buf, .. } => {
                        by_memory_target.entry(*buf).or_default().push(i);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        for group in by_memory_target.values() {
            for pair in group.windows(2) {
                edges.push((pair[0], pair[1]));
                in_degree[pair[1]] += 1;
            }
        }

        // Indices are emitted in ascending axis order.
        let mut index_positions: Vec<(u32, usize)> = rest
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| match self.at(id) {
                Op::Range { axis, .. } => Some((*axis, i)),
                _ => None,
            })
            .collect();
        index_positions.sort_by_key(|&(axis, _)| axis);
        for pair in index_positions.windows(2) {
            let (_, prev) = pair[0];
            let (_, next) = pair[1];
            edges.push((prev, next));
            in_degree[next] += 1;
        }

        // Control flow and barriers keep their mutual order.
        let mut prev_structural = usize::MAX;
        let mut structural_positions = Vec::with_capacity(structural.iter().filter(|b| **b).count());
        for i in 0..n {
            if structural[i] {
                structural_positions.push(i);
                if prev_structural != usize::MAX {
                    edges.push((prev_structural, i));
                    in_degree[i] += 1;
                }
                prev_structural = i;
            }
        }
        // Sinking prevention: an operation must never be emitted inside a
        // loop/if scope it was not already inside. Sinking a definition into a
        // deeper scope makes it invisible to uses in sibling/enclosing regions
        // and would undo LICM's hoisting.
        //
        // - A definition placed after a loop/if opener that starts after it
        //   would end up inside that scope, so it must stay before it.
        // - A definition must stay after any loop/if closer that precedes it,
        //   so it cannot be pulled backward into an already-closed scope.
        //
        // Both edge kinds point forward in the original order, so the original
        // order remains a valid topological order (no cycles). Hoisting toward
        // an enclosing scope stays allowed; only sinking is prevented.
        let mut openers = Vec::with_capacity(structural_positions.len());
        let mut closers = Vec::with_capacity(structural_positions.len());
        for &i in &structural_positions {
            match self.at(rest[i]) {
                Op::Loop { .. } => openers.push(i),
                Op::EndLoop => closers.push(i),
                _ => {}
            }
        }
        for i in 0..n {
            for &j in &openers {
                if i < j {
                    edges.push((i, j));
                    in_degree[j] += 1;
                }
            }
            for &j in &closers {
                if j < i {
                    edges.push((j, i));
                    in_degree[i] += 1;
                }
            }
        }

        // Hoisting prevention: a param/storage must never leave the scope it was
        // created in. Hoisting a register computation out of a loop breaks
        // per-iteration register reset semantics that downstream passes (e.g.
        // `merge`) rely on to keep nested reduce loops intact. So
        // a param/storage must stay after every opener that precedes it and before
        // every closer that follows it.
        for i in (0..n).filter(|&i| matches!(self.at(rest[i]), Op::Storage { .. })) {
            for &j in &openers {
                if j < i {
                    edges.push((j, i));
                    in_degree[i] += 1;
                }
            }
            for &j in &closers {
                if i < j {
                    edges.push((i, j));
                    in_degree[j] += 1;
                }
            }
        }

        // A group range is a parallel axis: nothing in the graph depends on
        // it, so the sort is free to hoist a per-group storage declaration or
        // its zeroing above it. A renderer that serializes the axis (the C
        // backend emits it as a loop) then runs that state setup once for the
        // whole grid instead of once per group. The axes may sit anywhere
        // during the passes; emission order is what the renderers see, so fix
        // each axis in place with every later op after it. This is not a
        // scope: hoisting toward the axis remains allowed for the ops that
        // already preceded it.
        //
        // The ranges' own length computations are exempt. A length may be
        // placed after its range by an earlier pass, and pinning it after the
        // range as well would close `range -> len -> range` into a cycle; the
        // cone is downward-closed along the operand edges, so exempting all
        // of it is exactly the set of ops a later op can reach the range
        // through.
        let range_positions: Vec<usize> = (0..n).filter(|&i| matches!(self.at(rest[i]), Op::Range { .. })).collect();
        let mut range_cone = vec![false; n];
        {
            let mut stack: Vec<usize> = range_positions.clone();
            while let Some(i) = stack.pop() {
                if range_cone[i] {
                    continue;
                }
                range_cone[i] = true;
                for p in self.at(rest[i]).parameters() {
                    if !p.is_null() {
                        let j = idx[p.0 as usize];
                        if j != usize::MAX {
                            stack.push(j);
                        }
                    }
                }
            }
        }
        for &i in &range_positions {
            for j in (i + 1)..n {
                if range_cone[j] {
                    continue;
                }
                edges.push((i, j));
                in_degree[j] += 1;
            }
        }

        // Stores never leave the loops/ifs that contain them and never cross
        // barriers: keep every store ordered with every structural op.
        for i in 0..n {
            if !store[i] {
                continue;
            }
            for &j in &structural_positions {
                if i < j {
                    edges.push((i, j));
                    in_degree[j] += 1;
                } else {
                    edges.push((j, i));
                    in_degree[i] += 1;
                }
            }
        }

        // A load of a storage that is also stored inside a loop must stay
        // inside that loop: it reads a loop-carried value (e.g. an accumulator
        // register). Hoisting such a load above the loop opener would replace
        // every iteration's load with the pre-loop value and silently break
        // the accumulation. Mirror of the store-pinning rule above, applied
        // only to loads whose target is stored in the enclosing loop.
        //
        // First match every loop opener with its closer.
        let mut loop_bounds: Vec<(usize, usize)> = Vec::new();
        {
            let mut open_stack: Vec<usize> = Vec::new();
            for &i in &structural_positions {
                match self.at(rest[i]) {
                    Op::Loop { .. } => open_stack.push(i),
                    Op::EndLoop => {
                        if let Some(opener) = open_stack.pop() {
                            loop_bounds.push((opener, i));
                        }
                    }
                    _ => {}
                }
            }
        }
        // Then collect, per storage, every loop that encloses a store to it.
        let mut storage_loops: Map<OpId, Vec<(usize, usize)>> = Map::default();
        {
            let mut open_stack: Vec<usize> = Vec::new();
            let mut closer_of: Map<usize, usize> = Map::default();
            for &(opener, closer) in &loop_bounds {
                closer_of.insert(opener, closer);
            }
            for (i, &op) in rest.iter().enumerate() {
                // Writes name their buffer directly. The lowered shared store
                // is the only write form the PTX path leaves: leaving it out
                // hides the buffer from the pinning below, and its loads then
                // escape the guard that bounds their index.
                let buf = match self.at(op) {
                    Op::Loop { .. } => {
                        open_stack.push(i);
                        continue;
                    }
                    Op::EndLoop => {
                        open_stack.pop();
                        continue;
                    }
                    Op::Store { dst, .. } | Op::Copy { dst, .. } | Op::TT(TTOp::Copy { dst, .. }) => *dst,
                    Op::PTX(p) => match p.as_ref() {
                        PTXOp::StShared { buf, .. } => *buf,
                        _ => continue,
                    },
                    _ => continue,
                };
                for &opener in &open_stack {
                    let bound = (opener, closer_of[&opener]);
                    let loops = storage_loops.entry(buf).or_default();
                    if !loops.contains(&bound) {
                        loops.push(bound);
                    }
                }
            }
        }
        // Pin affected loads inside every enclosing loop that stores their
        // target. A load is inside a loop iff `opener < i < closer` in the
        // original order; the edges keep it there.
        for i in 0..n {
            if !load[i] {
                continue;
            }
            let buf = match self.at(rest[i]) {
                Op::Load { src, .. } => Some(*src),
                // `ldmatrix` reads its storage like a load; the fill that
                // writes it inside the loop makes it loop-carried.
                Op::PTX(p) => match p.as_ref() {
                    PTXOp::VLdMatrix { src, .. } => Some(*src),
                    PTXOp::LdShared { buf, .. } => Some(*buf),
                    // The lowered `ldmatrix` names a generic address, not the
                    // storage, so the loop it belongs to cannot be named here:
                    // pin it inside every loop that encloses it, which is the
                    // same rule with a wider set of candidates.
                    PTXOp::LdMatrix { .. } => None,
                    PTXOp::Bytes(_)
                    | PTXOp::WarpSync
                    | PTXOp::Shfl { .. }
                    | PTXOp::CvtaShared { .. }
                    | PTXOp::SharedBase { .. }
                    | PTXOp::StShared { .. } => continue,
                },
                _ => continue,
            };
            match buf {
                None => {
                    for &(opener, closer) in &loop_bounds {
                        if opener < i && i < closer {
                            edges.push((opener, i));
                            in_degree[i] += 1;
                            edges.push((i, closer));
                            in_degree[closer] += 1;
                        }
                    }
                }
                Some(buf) => {
                    let Some(loops) = storage_loops.get(&buf) else { continue };
                    for &(opener, closer) in loops {
                        if opener < i && i < closer {
                            edges.push((opener, i));
                            in_degree[i] += 1;
                            edges.push((i, closer));
                            in_degree[closer] += 1;
                        }
                    }
                }
            }
        }

        // A load from Circular storage is a FIFO pop: hoisting it out of
        // any enclosing loop changes pop count and position even when no
        // store to it sits in that loop (cross-section streaming, e.g. a
        // reader producing per-iteration while compute consumes
        // per-iteration). Pin such loads inside their full original loop
        // nest; the loop-carried rule above only covers loops that also
        // store to the target.
        for i in 0..n {
            if !load[i] {
                continue;
            }
            // A load from Circular storage is a FIFO pop (see above):
            // the base names the buffer. A copy reading from Circular
            // pops the same way and pins identically.
            let buf = match self.at(rest[i]) {
                Op::Load { src, .. } | Op::Copy { src, .. } | Op::TT(TTOp::Copy { src, .. }) => *src,
                _ => continue,
            };
            if !matches!(self.at(buf), Op::Storage { scope: MemScope::Circular, .. }) {
                continue;
            }
            for &(opener, closer) in &loop_bounds {
                if opener < i && i < closer {
                    edges.push((opener, i));
                    in_degree[i] += 1;
                    edges.push((i, closer));
                    in_degree[closer] += 1;
                }
            }
        }

        let barrier_positions: Vec<usize> = (0..n).filter(|&i| barrier[i]).collect();
        // Loads never cross barriers.
        for i in 0..n {
            if !load[i] {
                continue;
            }
            for &j in &barrier_positions {
                if i < j {
                    edges.push((i, j));
                    in_degree[j] += 1;
                } else {
                    edges.push((j, i));
                    in_degree[i] += 1;
                }
            }
        }

        // Loads keep their original order with inline asm: asm is an
        // opaque side effect (e.g. datapath setup must precede the
        // first copy), so traffic must not float across it.
        let asm_positions: Vec<usize> = (0..n).filter(|&i| matches!(self.at(rest[i]), Op::Asm { .. })).collect();
        for i in 0..n {
            if !load[i] {
                continue;
            }
            for &j in &asm_positions {
                if i < j {
                    edges.push((i, j));
                    in_degree[j] += 1;
                } else {
                    edges.push((j, i));
                    in_degree[i] += 1;
                }
            }
        }

        // Lay the collected edges out in CSR form: `offsets[i]..offsets[i+1]`
        // indexes `targets` with the dependents of operation `i`.
        let mut offsets = vec![0usize; n + 1];
        for &(u, _) in &edges {
            offsets[u + 1] += 1;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut fill: Vec<usize> = offsets[..n].to_vec();
        let mut targets = vec![0usize; edges.len()];
        for &(u, v) in &edges {
            targets[fill[u]] = v;
            fill[u] += 1;
        }

        // Height of every op: the longest latency-weighted path from it to a
        // sink. Every precedence edge points forward in the original order (see
        // the rules above), so the original order is already a topological
        // order and one reverse sweep computes all heights.
        let mut height = vec![0u64; n];
        for i in (0..n).rev() {
            let mut h = 0u64;
            for &j in &targets[offsets[i]..offsets[i + 1]] {
                h = h.max(height[j]);
            }
            height[i] = h + latency[i];
        }
        // How many consumers each op still has to be emitted for. At zero the op
        // has been consumed, which is what frees a capped slot below.
        let mut users = vec![0u32; n];
        for i in 0..n {
            for &p in &params_flat[param_start[i]..param_start[i + 1]] {
                users[p] += 1;
            }
        }

        // Latency-ordered topological sort: among ready operations, emit the
        // one on the longest path to a sink, so the work that a long-latency
        // result unblocks starts as early as its inputs allow. Read-only
        // memory ops are the longest paths in a GEMM body, which is what puts
        // the next chunk's global loads at the top of the loop; ties break by
        // original position, so the result is deterministic.
        //
        // A shared fragment load holds its result in registers until its
        // consumer runs, so only `FRAGMENT_SLOTS` of them may be in flight:
        // issuing four and letting their consumers drain is what makes the
        // body alternate fragment loads with the tensor-core work that uses
        // them, instead of reading the whole tile before the first `mma`.
        const FRAGMENT_SLOTS: usize = 4;
        let mut in_flight = 0usize;
        let mut deferred: Vec<usize> = Vec::new();
        let mut ready: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                ready.push(Reverse((usize::MAX - height[i] as usize, i)));
            }
        }
        let mut result = Vec::with_capacity(n);
        while result.len() < n {
            // Pop until an op is selectable: a capped fragment load that finds
            // every slot taken waits for the next round instead of forcing the
            // register file past its budget.
            let mut chosen = None;
            while let Some(Reverse((_, i))) = ready.pop() {
                if frag_load[i] && in_flight >= FRAGMENT_SLOTS {
                    deferred.push(i);
                    continue;
                }
                chosen = Some(i);
                break;
            }
            // A capped op is emitted anyway when nothing else is ready: the cap
            // orders the body, it must never stall or reorder it into a cycle.
            let Some(i) = chosen.or_else(|| (!deferred.is_empty()).then(|| deferred.remove(0))) else {
                break;
            };
            for j in deferred.drain(..) {
                ready.push(Reverse((usize::MAX - height[j] as usize, j)));
            }
            if frag_load[i] && users[i] > 0 {
                in_flight += 1;
            }
            result.push(rest[i]);
            for &p in &params_flat[param_start[i]..param_start[i + 1]] {
                users[p] -= 1;
                if users[p] == 0 && frag_load[p] {
                    in_flight -= 1;
                }
            }
            for &j in &targets[offsets[i]..offsets[i + 1]] {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    ready.push(Reverse((usize::MAX - height[j] as usize, j)));
                }
            }
        }
        debug_assert_eq!(result.len(), n, "cycle in instruction dependencies");
        result
    }
}
