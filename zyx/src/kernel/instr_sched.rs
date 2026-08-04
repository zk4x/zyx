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

use crate::{
    Map, Set,
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

        let pos: Map<OpId, usize> = rest.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let sorted_rest = self.schedule_rest(&rest, &pos);

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
    ///
    /// Everything else is free to move; among the ready operations those that
    /// depend on the fewest other operations are emitted first, with ties
    /// broken by original position.
    fn schedule_rest(&self, rest: &[OpId], pos: &Map<OpId, usize>) -> Vec<OpId> {
        let in_rest: Set<OpId> = rest.iter().copied().collect();
        let mut dependents: Map<OpId, Vec<OpId>> = Map::default();
        let mut n_params: Map<OpId, usize> = Map::default();

        let mut structural: Vec<OpId> = Vec::new();
        let mut barriers: Vec<OpId> = Vec::new();
        let mut stores: Vec<OpId> = Vec::new();
        let mut loads: Vec<OpId> = Vec::new();

        for &id in rest {
            let params: Vec<OpId> = self.at(id).parameters().collect();
            n_params.insert(id, params.len());
            for p in params {
                if in_rest.contains(&p) {
                    dependents.entry(p).or_default().push(id);
                }
            }
            match self.at(id) {
                Op::Barrier => {
                    structural.push(id);
                    barriers.push(id);
                }
                Op::Loop { .. } | Op::EndLoop | Op::If { .. } | Op::EndIf => structural.push(id),
                Op::Store { .. } => stores.push(id),
                Op::Load { .. } => loads.push(id),
                _ => {}
            }
        }

        // Loads and stores to the same define keep their relative order.
        let mut by_define: Map<OpId, Vec<OpId>> = Map::default();
        for &id in rest {
            match self.at(id) {
                Op::Load { src, .. } => by_define.entry(*src).or_default().push(id),
                Op::Store { dst, .. } => by_define.entry(*dst).or_default().push(id),
                _ => {}
            }
        }
        for group in by_define.values() {
            for pair in group.windows(2) {
                dependents.entry(pair[0]).or_default().push(pair[1]);
            }
        }

        // Stores never leave their loop/if and never cross barriers: keep
        // every store ordered with every structural op.
        for &store in &stores {
            for &structural in &structural {
                if pos[&store] < pos[&structural] {
                    dependents.entry(store).or_default().push(structural);
                } else {
                    dependents.entry(structural).or_default().push(store);
                }
            }
        }
        // Loads never cross barriers.
        for &load in &loads {
            for &barrier in &barriers {
                if pos[&load] < pos[&barrier] {
                    dependents.entry(load).or_default().push(barrier);
                } else {
                    dependents.entry(barrier).or_default().push(load);
                }
            }
        }

        let mut in_degree: Map<OpId, usize> = rest.iter().map(|&id| (id, 0)).collect();
        for (_, ds) in &dependents {
            for &d in ds {
                *in_degree.get_mut(&d).unwrap() += 1;
            }
        }

        let mut ready: Vec<OpId> = rest.iter().copied().filter(|id| in_degree[id] == 0).collect();
        let mut result = Vec::with_capacity(rest.len());
        let mut emitted = 0;
        while emitted < rest.len() {
            let next = ready
                .iter()
                .min_by_key(|id| (n_params[id], pos[id]))
                .copied()
                .expect("cycle in instruction dependencies");
            ready.retain(|id| *id != next);
            result.push(next);
            emitted += 1;
            if let Some(ds) = dependents.get(&next) {
                for &d in ds {
                    let degree = in_degree.get_mut(&d).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push(d);
                    }
                }
            }
        }
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