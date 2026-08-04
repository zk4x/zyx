// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Instruction scheduling for kernel optimization.
//!
//! This module provides instruction scheduling optimizations for kernels,
//! including:
//!
//! - Ordering defines (global read-only, global read-write, local read-only,
//!   local read-write) at the beginning of the kernel.
//! - Topologically sorting the remaining operations by dependency so that
//!   operations which depend on the fewest other operations come first.
//! - Improving instruction pipeline utilization.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::{
    Map,
    kernel::{Kernel, MemScope, Op, OpId},
};

impl Kernel {
    /// Schedule instructions for better instruction throughput.
    ///
    /// This method reorders kernel operations to improve instruction
    /// scheduling. The final order is:
    ///
    /// 1. All global read-only defines, preserving their order.
    /// 2. All global read-write defines, preserving their order.
    /// 3. All local read-only defines, preserving their order.
    /// 4. All local read-write defines, preserving their order.
    /// 5. The remaining operations, topologically sorted by dependency with
    ///    operations that depend on the fewest other operations first.
    ///
    /// Memory operations that share a define keep their relative order, stores
    /// are never moved out of the loops or if blocks that contain them, and
    /// stores and loads are never moved before barriers.
    pub fn instruction_schedule(&mut self) {
        let mut global_ro = Vec::new();
        let mut global_rw = Vec::new();
        let mut local_ro = Vec::new();
        let mut local_rw = Vec::new();
        let mut rest = Vec::new();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.at(op_id) {
                Op::Define { scope: MemScope::Global, ro: true, .. } => global_ro.push(op_id),
                Op::Define { scope: MemScope::Global, ro: false, .. } => global_rw.push(op_id),
                Op::Define { scope: MemScope::Local, ro: true, .. } => local_ro.push(op_id),
                Op::Define { scope: MemScope::Local, ro: false, .. } => local_rw.push(op_id),
                _ => rest.push(op_id),
            }
            op_id = next;
        }

        let sorted_rest = self.schedule_rest(&rest);

        let mut order =
            Vec::with_capacity(global_ro.len() + global_rw.len() + local_ro.len() + local_rw.len() + sorted_rest.len());
        order.extend(global_ro);
        order.extend(global_rw);
        order.extend(local_ro);
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

    /// Topologically sort the non-define operations by dependency.
    ///
    /// The sort respects the following precedence constraints:
    ///
    /// - A use must come after its declaration.
    /// - Loads and stores that share a define keep their relative order, so
    ///   memory hazards (RAW, WAR, WAW) on the same buffer are preserved.
    /// - Stores never leave the loops or if blocks that contain them and never
    ///   cross barriers.
    /// - Loads never cross barriers.
    /// - Control flow and barriers keep their mutual order.
    ///
    /// Everything else is free to move; among the ready operations those that
    /// depend on the fewest other operations are emitted first, with ties
    /// broken by original position. Runs in O(n log n + e) where `e` is the
    /// number of precedence edges.
    fn schedule_rest(&self, rest: &[OpId]) -> Vec<OpId> {
        let n = rest.len();
        let idx: Map<OpId, usize> = rest.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let mut structural = vec![false; n];
        let mut barrier = vec![false; n];
        let mut store = vec![false; n];
        let mut load = vec![false; n];
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Barrier => {
                    structural[i] = true;
                    barrier[i] = true;
                }
                Op::Loop { .. } | Op::EndLoop | Op::If { .. } | Op::EndIf => structural[i] = true,
                Op::Store { .. } => store[i] = true,
                Op::Load { .. } => load[i] = true,
                _ => {}
            }
        }

        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];
        let mut n_params = vec![0usize; n];

        // Uses must come after their declarations.
        for (i, &id) in rest.iter().enumerate() {
            let params: Vec<OpId> = self.at(id).parameters().collect();
            n_params[i] = params.len();
            for p in params {
                if let Some(&j) = idx.get(&p) {
                    dependents[j].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        // Loads and stores to the same define keep their relative order.
        let mut by_define: Map<OpId, Vec<usize>> = Map::default();
        for (i, &id) in rest.iter().enumerate() {
            match self.at(id) {
                Op::Load { src, .. } => by_define.entry(*src).or_default().push(i),
                Op::Store { dst, .. } => by_define.entry(*dst).or_default().push(i),
                _ => {}
            }
        }
        for group in by_define.values() {
            for pair in group.windows(2) {
                dependents[pair[0]].push(pair[1]);
                in_degree[pair[1]] += 1;
            }
        }

        // Control flow and barriers keep their mutual order.
        let mut prev_structural = usize::MAX;
        for i in 0..n {
            if structural[i] {
                if prev_structural != usize::MAX {
                    dependents[prev_structural].push(i);
                    in_degree[i] += 1;
                }
                prev_structural = i;
            }
        }

        // Stores never leave the loops/ifs that contain them and never cross
        // barriers: keep every store ordered with every structural op.
        for i in 0..n {
            if !store[i] {
                continue;
            }
            for j in 0..n {
                if !structural[j] {
                    continue;
                }
                if i < j {
                    dependents[i].push(j);
                    in_degree[j] += 1;
                } else {
                    dependents[j].push(i);
                    in_degree[i] += 1;
                }
            }
        }
        // Loads never cross barriers.
        for i in 0..n {
            if !load[i] {
                continue;
            }
            for j in 0..n {
                if !barrier[j] {
                    continue;
                }
                if i < j {
                    dependents[i].push(j);
                    in_degree[j] += 1;
                } else {
                    dependents[j].push(i);
                    in_degree[i] += 1;
                }
            }
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
            for &j in &dependents[i] {
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

#[cfg(test)]
mod tests {
    use crate::DType;
    use crate::kernel::{DeviceId, Kernel, MemLayout, MemScope, Op, OpId};

    fn defines_in_order(k: &Kernel) -> Vec<(MemScope, bool)> {
        let mut order = Vec::new();
        let mut op_id = k.head;
        while !op_id.is_null() {
            if let Op::Define { scope, ro, .. } = k.at(op_id) {
                order.push((*scope, *ro));
            }
            op_id = k.next_op(op_id);
        }
        order
    }

    fn op_ids_in_order(k: &Kernel) -> Vec<OpId> {
        let mut order = Vec::new();
        let mut op_id = k.head;
        while !op_id.is_null() {
            order.push(op_id);
            op_id = k.next_op(op_id);
        }
        order
    }

    #[test]
    fn test_instruction_schedule_orders_defines() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let _local_rw = k.define(DType::F32, MemScope::Local, false, 4);
        let global_ro = k.define(DType::F32, MemScope::Global, true, 4);
        let _local_ro = k.define(DType::F32, MemScope::Local, true, 4);
        let global_rw = k.define(DType::F32, MemScope::Global, false, 4);

        let gidx = k.group_index(0, 4);
        let c = k.const_val(1.0f32);
        let load = k.load(global_ro, gidx, MemLayout::Scalar);
        let add = k.add(load, c);
        k.store(global_rw, add, gidx, MemLayout::Scalar);

        k.instruction_schedule();

        assert_eq!(
            defines_in_order(&k),
            vec![
                (MemScope::Global, true),
                (MemScope::Global, false),
                (MemScope::Local, true),
                (MemScope::Local, false),
            ]
        );

        let order = op_ids_in_order(&k);
        let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
        assert!(pos(gidx) < pos(load));
        assert!(pos(c) < pos(add));
        assert!(pos(load) < pos(add));
    }

    #[test]
    fn test_instruction_schedule_keeps_stores_in_loops() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let src = k.define(DType::F32, MemScope::Global, true, 4);
        let dst = k.define(DType::F32, MemScope::Global, false, 4);

        let len = k.const_idx(4u32);
        let loop_id = k.loop_(len);
        let in_loop_load = k.load(src, loop_id, MemLayout::Scalar);
        let add = k.add(in_loop_load, in_loop_load);
        k.store(dst, add, loop_id, MemLayout::Scalar);
        k.end_loop();

        k.instruction_schedule();

        let order = op_ids_in_order(&k);
        let store = order.iter().copied().find(|&id| matches!(k.at(id), Op::Store { .. })).unwrap();
        let end_loop = order.iter().copied().find(|&id| matches!(k.at(id), Op::EndLoop)).unwrap();
        let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
        assert!(pos(loop_id) < pos(store), "store must stay inside its loop");
        assert!(pos(store) < pos(end_loop), "store must stay inside its loop");
    }

    #[test]
    fn test_instruction_schedule_keeps_memory_order_per_define() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let buf = k.define(DType::F32, MemScope::Global, false, 4);

        let gidx = k.group_index(0, 4);
        let val = k.const_val(1.0f32);
        k.store(buf, val, gidx, MemLayout::Scalar);
        let load = k.load(buf, gidx, MemLayout::Scalar);
        k.store(buf, load, gidx, MemLayout::Scalar);

        k.instruction_schedule();

        let order = op_ids_in_order(&k);
        let stores: Vec<OpId> = order.iter().copied().filter(|&id| matches!(k.at(id), Op::Store { .. })).collect();
        let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
        assert_eq!(stores.len(), 2);
        assert!(pos(stores[0]) < pos(load), "load to a define must stay after prior store to it");
        assert!(pos(load) < pos(stores[1]), "load to a define must stay before later store to it");
    }

    #[test]
    fn test_instruction_schedule_keeps_stores_after_barriers() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let buf = k.define(DType::F32, MemScope::Local, false, 4);

        let gidx = k.group_index(0, 4);
        let val = k.const_val(1.0f32);
        k.barrier();
        k.store(buf, val, gidx, MemLayout::Scalar);

        k.instruction_schedule();

        let order = op_ids_in_order(&k);
        let store = order.iter().copied().find(|&id| matches!(k.at(id), Op::Store { .. })).unwrap();
        let barrier = order.iter().copied().find(|&id| matches!(k.at(id), Op::Barrier)).unwrap();
        let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
        assert!(pos(barrier) < pos(store), "store must stay after the barrier");
    }

    #[test]
    fn test_instruction_schedule_topological() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let src = k.define(DType::F32, MemScope::Global, true, 4);
        let dst = k.define(DType::F32, MemScope::Global, false, 4);

        let gidx = k.group_index(0, 4);
        let a = k.load(src, gidx, MemLayout::Scalar);
        let b = k.load(src, gidx, MemLayout::Scalar);
        let add = k.add(a, b);
        k.store(dst, add, gidx, MemLayout::Scalar);

        k.instruction_schedule();

        let order = op_ids_in_order(&k);
        let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
        assert!(pos(a) < pos(add));
        assert!(pos(b) < pos(add));
    }
}