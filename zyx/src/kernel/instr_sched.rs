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
    kernel::{Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp},
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
        let mut global_ro = Vec::new();
        let mut global_rw = Vec::new();
        let mut local_rw = Vec::new();
        let mut rest = Vec::new();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.at(op_id) {
                Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } => global_ro.push(op_id),
                Op::Param { kind: ParamKind::GlobalMut, .. } => global_rw.push(op_id),
                Op::Storage { scope: MemScope::Local, .. } => local_rw.push(op_id),
                _ => rest.push(op_id),
            }
            op_id = next;
        }

        let sorted_rest = self.schedule_rest(&rest);

        let mut order = Vec::with_capacity(global_ro.len() + global_rw.len() + local_rw.len() + sorted_rest.len());
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
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Barrier | Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => {
                    structural[i] = true;
                    barrier[i] = true;
                }
                Op::Loop { .. } | Op::EndLoop => structural[i] = true,
                // Opaque side effect (inline asm, TT LLK call):
                // program-ordered like a barrier for placement (never
                // crosses one), but not a barrier itself.
                Op::Asm { .. } | Op::TT(TTOp::LLK { .. }) => structural[i] = true,
                Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => structural[i] = true,
                Op::Store { .. } => store[i] = true,
                Op::Load { .. } => load[i] = true,
                // A copy reads its src and writes its dst: it pins like a
                // load (barriers, asm, loops) and orders like a store.
                Op::Copy { .. } => {
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
        let mut n_params = vec![0usize; n];

        // Uses must come after their declarations. Iterate params directly
        // without collecting into a Vec.
        for (i, &id) in rest.iter().enumerate() {
            let mut count = 0usize;
            macro_rules! add_param {
                ($p:expr) => {{
                    count += 1;
                    let j = idx[$p.0 as usize];
                    if j != usize::MAX {
                        edges.push((j, i));
                        in_degree[i] += 1;
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
                Op::TT(TTOp::ReduceTile { x, scaler, acc, .. }) => {
                    add_param!(x);
                    add_param!(scaler);
                    add_param!(acc);
                }
                Op::TT(TTOp::MatmulTile { x, y, acc }) => {
                    add_param!(x);
                    add_param!(y);
                    add_param!(acc);
                }
                Op::TT(TTOp::TransposeTile { x }) => add_param!(x),
                Op::TT(TTOp::BroadcastTile { x, .. }) => add_param!(x),
                Op::TT(TTOp::ReserveBack { cb, .. })
                | Op::TT(TTOp::PushBack { cb, .. })
                | Op::TT(TTOp::WaitFront { cb, .. })
                | Op::TT(TTOp::PopFront { cb, .. }) => add_param!(cb),
                Op::Binary { x, y, .. } => {
                    add_param!(x);
                    add_param!(y);
                }
                Op::Param { .. }
                | Op::Const(_)
                | Op::Storage { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute) => {}
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Kernel { .. } | Op::Custom(_) => {
                    todo!()
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => add_param!(len),
                    RangeKind::Warp(local_id) => add_param!(local_id),
                    RangeKind::Local(_) => {}
                },
                Op::Store { dst, src: x } => {
                    add_param!(dst);
                    add_param!(x);
                }
                Op::Copy { src, dst } => {
                    add_param!(src);
                    add_param!(dst);
                }
                Op::Load { src } => {
                    add_param!(src);
                }
                Op::GEP { x, index, .. } => {
                    add_param!(x);
                    add_param!(index);
                }
                Op::Loop { len, .. } => add_param!(len),
                Op::Mad { x, y, z } => {
                    add_param!(x);
                    add_param!(y);
                    add_param!(z);
                }
                Op::Wmma { a, b, c, .. } => {
                    add_param!(a);
                    add_param!(b);
                    add_param!(c);
                }
                Op::Asm { ops, .. } | Op::TT(TTOp::LLK { ops, .. }) => {
                    for &p in ops.iter() {
                        add_param!(p);
                    }
                }
                Op::TT(TTOp::LLKReduce { cb_in, cb_sc, slot, x, scaler, .. }) => {
                    add_param!(cb_in);
                    add_param!(cb_sc);
                    add_param!(slot);
                    add_param!(x);
                    add_param!(scaler);
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    add_param!(cb_a);
                    add_param!(cb_b);
                    add_param!(mx);
                    add_param!(plain);
                }
                Op::Stack { ops } => {
                    for &p in ops.iter() {
                        add_param!(p);
                    }
                }
                Op::Index { vec, .. } => add_param!(vec),
            }
            n_params[i] = count;
        }

        // Loads, stores, and copies to the same buffer keep their relative
        // order. Locations are GEPs (or bare Params pre-linearize): group
        // by the addressed buffer, not the GEP id, or same-buffer traffic
        // through different GEPs could reorder across FIFO accounting.
        let mut by_memory_target: Map<OpId, Vec<usize>> = Map::default();
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Load { src, .. } => {
                    let buf = match self.at(*src) {
                        Op::GEP { x, .. } => *x,
                        _ => *src,
                    };
                    by_memory_target.entry(buf).or_default().push(i);
                }
                Op::Store { dst, .. } => {
                    let buf = match self.at(*dst) {
                        Op::GEP { x, .. } => *x,
                        _ => *dst,
                    };
                    by_memory_target.entry(buf).or_default().push(i);
                }
                Op::Copy { src, dst } => {
                    for &side in &[src, dst] {
                        let buf = match self.at(*side) {
                            Op::GEP { x, .. } => *x,
                            _ => *side,
                        };
                        by_memory_target.entry(buf).or_default().push(i);
                    }
                }
                // CB sync ops name their buffer directly (no GEP): same
                // FIFO group as the CB's loads/stores/copies, or sync
                // could reorder across the traffic it guards.
                Op::TT(TTOp::ReserveBack { cb, .. })
                | Op::TT(TTOp::PushBack { cb, .. })
                | Op::TT(TTOp::WaitFront { cb, .. })
                | Op::TT(TTOp::PopFront { cb, .. }) => {
                    by_memory_target.entry(*cb).or_default().push(i);
                }
                // LLK calls stream CBs (compute) or publish into them
                // (pack): same FIFO group as the CB's traffic, or the
                // call could reorder across the sync it runs under.
                // Index operands form singleton groups (harmless).
                Op::TT(TTOp::LLK { ops, .. }) => {
                    for &x in ops.iter() {
                        by_memory_target.entry(x).or_default().push(i);
                    }
                }
                Op::TT(TTOp::LLKReduce { cb_in, cb_sc, slot, x, scaler, .. }) => {
                    for &f in [cb_in, cb_sc, slot, x, scaler] {
                        by_memory_target.entry(f).or_default().push(i);
                    }
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    for &f in [cb_a, cb_b, mx, plain] {
                        by_memory_target.entry(f).or_default().push(i);
                    }
                }
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
        // defined in. Hoisting a register define out of a loop breaks
        // per-iteration register reset semantics that downstream passes (e.g.
        // `merge_nested_loops`) rely on to keep nested reduce loops intact. So
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
            for i in 0..n {
                match self.at(rest[i]) {
                    Op::Loop { .. } => open_stack.push(i),
                    Op::EndLoop => {
                        open_stack.pop();
                    }
                    Op::Store { dst, .. } | Op::Copy { dst, .. } => {
                        // Writes address a GEP (or a bare Param
                        // pre-linearize): track the buffer, not the GEP.
                        let buf = match self.at(*dst) {
                            Op::GEP { x, .. } => *x,
                            _ => *dst,
                        };
                        for &opener in &open_stack {
                            let bound = (opener, closer_of[&opener]);
                            let loops = storage_loops.entry(buf).or_default();
                            if !loops.contains(&bound) {
                                loops.push(bound);
                            }
                        }
                    }
                    _ => {}
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
            let Op::Load { src, .. } = self.at(rest[i]) else { continue };
            let buf = match self.at(*src) {
                Op::GEP { x, .. } => *x,
                _ => *src,
            };
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
            // A load from Circular storage is a FIFO pop (see above): chase
            // the load's GEP to the buffer. A copy reading from Circular
            // pops the same way and pins identically.
            let src = match self.at(rest[i]) {
                Op::Load { src, .. } | Op::Copy { src, .. } => *src,
                _ => continue,
            };
            let buf = match self.at(src) {
                Op::GEP { x, .. } => *x,
                _ => src,
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

        // Stable topological sort: among ready operations, emit the one that
        // depends on the fewest other operations first, ties broken by
        // original position.
        let mut ready: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                ready.push(Reverse((n_params[i], i)));
            }
        }
        let mut result = Vec::with_capacity(n);
        while let Some(Reverse((_, i))) = ready.pop() {
            result.push(rest[i]);
            for &j in &targets[offsets[i]..offsets[i + 1]] {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    ready.push(Reverse((n_params[j], j)));
                }
            }
        }
        debug_assert_eq!(result.len(), n, "cycle in instruction dependencies");
        result
    }
}
