// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxScope, Kernel, MemLayout, MemScope, Op, OpId},
    shape::Dim,
};

fn gather_deps(kernel: &Kernel, seeds: &[OpId]) -> Set<OpId> {
    let mut visited = Set::default();
    let mut stack: Vec<OpId> = seeds.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        stack.extend(kernel.ops[id].op.parameters());
    }
    visited
}

fn round_up(len: Dim, multiple: Dim) -> Dim {
    let rem = len % multiple;
    if rem == 0 { 0 } else { multiple - rem }
}

impl Kernel {
    pub(crate) fn opt_tenstorrent_tile(&mut self) {
        if self.ops.values().any(|node| matches!(node.op, Op::Loop { .. })) {
            self.tenstorrent_reduce_pad();
            // orig_len is the reduce dim length before padding to 32
            self.pad_loop();
        } else {
            self.tenstorrent_pad();
            self.tenstorrent_local();
            self.tenstorrent_group();
            self.tenstorrent_loop_local();
        }
    }

    fn tenstorrent_reduce_pad(&mut self) {
        let mut gidxs: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::Index { len, axis, scope } = self.at(op_id) {
                if scope == IdxScope::Group {
                    gidxs.push((op_id, axis, len));
                } else {
                    return;
                }
            }
            op_id = self.next_op(op_id);
        }
        gidxs.sort_by_key(|&(_, axis, _)| axis);
        gidxs.dedup_by_key(|&mut (_, axis, _)| axis);

        for &(id, _axis, len) in &gidxs {
            let pad = round_up(len, 32);
            if pad > 0 {
                self.pad_index(id, pad);
            }
        }

        self.verify();
    }

    fn tenstorrent_pad(&mut self) {
        let mut gidxs: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::Index { len, axis, scope } = self.at(op_id) {
                if scope == IdxScope::Group {
                    gidxs.push((op_id, axis, len));
                } else {
                    // Can't run this optimization on kernel that already has local indices
                    continue;
                }
            }
            op_id = self.next_op(op_id);
        }
        gidxs.sort_by_key(|&(_, axis, _)| axis);
        gidxs.dedup_by_key(|&mut (_, axis, _)| axis);

        match gidxs.len() {
            0 | 2 => {
                for &(id, _axis, len) in &gidxs {
                    let pad = round_up(len, 32);
                    if pad > 0 {
                        self.pad_index(id, pad);
                    }
                }
            }
            1 => {
                let (id, _axis, len) = gidxs[0];
                let pad = round_up(len, 1024);
                if pad > 0 {
                    self.pad_index(id, pad);
                }
                let new_len = if let Op::Index { len, scope: IdxScope::Group, .. } = self.at(id) {
                    *len
                } else {
                    unreachable!()
                };
                let sqrt = (new_len as f64).sqrt() as Dim;
                let f1 = (32..=sqrt).rev().find(|&f| f % 32 == 0 && new_len % f == 0);
                let Some(f1) = f1 else {
                    return;
                };
                let f2 = new_len / f1;
                self.split_dim(
                    id,
                    vec![
                        Op::Index { len: f1, axis: 0, scope: IdxScope::Group },
                        Op::Index { len: f2, axis: 1, scope: IdxScope::Group },
                    ],
                );
            }
            3 => {
                let (last_id, _last_axis, last_len) = gidxs[2];
                for &(id, _axis, len) in &gidxs[..2] {
                    let pad = round_up(len, 32);
                    if pad > 0 {
                        self.pad_index(id, pad);
                    }
                }
                let len_const = self.insert_before(last_id, Op::Const(Constant::idx(last_len)));
                self.ops[last_id].op = Op::Loop { len: len_const };
                self.push_back(Op::EndLoop);
            }
            _ => {}
        }

        self.verify();
    }

    fn tenstorrent_local(&mut self) {
        // Step 1: Split each GroupIndex into GroupIndex(len/32) + Loop(32)
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, axis, scope: IdxScope::Group } = self.ops[op_id].op {
                if len % 32 == 0 && len >= 32 {
                    let f1 = len / 32;
                    self.split_dim(
                        op_id,
                        vec![
                            Op::Index { len: f1, axis, scope: IdxScope::Group },
                            Op::Index { len: 32, axis, scope: IdxScope::Local },
                        ],
                    );
                } else {
                    return;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Step 2: Verify exactly 2 local indices of len 32 (axes 0 and 1)
        let mut lidxs = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, axis, scope: IdxScope::Local } = self.at(op_id) {
                if *len != 32 {
                    return;
                }
                lidxs.push((*axis, op_id));
            }
            op_id = self.next_op(op_id);
        }
        if lidxs.len() != 2 {
            return;
        }
        lidxs.sort();
        if lidxs[0].0 != 0 || lidxs[1].0 != 1 {
            return;
        }
        let lidx0 = lidxs[0].1;
        let lidx1 = lidxs[1].1;

        // Step 3: Find all scalar loads from global defines
        let global_loads: Vec<(OpId, OpId, OpId)> = {
            let mut loads = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Load { src, index, layout: MemLayout::Scalar } = self.at(op_id) {
                    if matches!(self.at(*src), Op::Define { scope: MemScope::Global, .. }) {
                        loads.push((op_id, *src, *index));
                    }
                }
                op_id = self.next_op(op_id);
            }
            loads
        };

        if global_loads.is_empty() {
            return;
        }

        // Step 4: Insert constant/index computation before the first load
        let first_load = global_loads[0].0;
        let first_global_idx = global_loads[0].2;
        let const_32 = self.insert_before(first_load, Op::Const(Constant::idx(32u32)));
        let scaled = self.insert_before(first_load, Op::Binary { x: lidx0, y: const_32, bop: BOp::Mul });
        let combined_idx = self.insert_before(first_load, Op::Binary { x: scaled, y: lidx1, bop: BOp::Add });
        let zero = self.insert_before(first_load, Op::Const(Constant::idx(0u32)));

        // Step 5: Find the last global define to insert locals after it
        let mut last_global = self.head;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Define { scope: MemScope::Global, .. }) {
                last_global = scan;
            }
            scan = self.next_op(scan);
        }

        // Step 6: Allocate local buffers for each unique global source
        let mut src_to_local: Map<OpId, OpId> = Map::default();
        for &(_, src, _) in &global_loads {
            if src_to_local.contains_key(&src) {
                continue;
            }
            let local = self
                .insert_after(last_global, Op::Define { dtype: self.dtype(src), scope: MemScope::Local, ro: false, len: 1024 });
            last_global = local;
            src_to_local.insert(src, local);
        }

        // Step 7: Insert all global→local stores before the first load, then a barrier.
        // Load from global at the first load's index (all element-wise loads use the same position),
        // store to local at the local tile index (lidx0*32 + lidx1).
        let mut processed: Set<OpId> = Set::default();
        for &(_, src, _) in &global_loads {
            if processed.insert(src) {
                let local = src_to_local[&src];
                let global_load =
                    self.insert_before(first_load, Op::Load { src, index: first_global_idx, layout: MemLayout::Scalar });
                self.insert_before(
                    first_load,
                    Op::Store { dst: local, x: global_load, index: combined_idx, layout: MemLayout::Scalar },
                );
            }
        }
        self.insert_before(first_load, Op::Barrier);

        // Step 8: Replace all original loads with tiled loads from local
        for &(load_op, src, _) in &global_loads {
            let local = src_to_local[&src];
            self.ops[load_op].op = Op::Load { src: local, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        // Step 9: Find all scalar stores to global defines
        let global_stores: Vec<(OpId, OpId, OpId, OpId)> = {
            let mut stores = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Store { dst, x, index, layout: MemLayout::Scalar } = self.at(op_id) {
                    if matches!(self.at(*dst), Op::Define { scope: MemScope::Global, .. }) {
                        stores.push((op_id, *dst, *x, *index));
                    }
                }
                op_id = self.next_op(op_id);
            }
            stores
        };

        if global_stores.is_empty() {
            return;
        }

        // Step 10: Allocate local buffers for each unique global destination
        let mut dst_to_local: Map<OpId, OpId> = Map::default();
        let mut last_local = last_global;
        for &(_, dst, _, _) in &global_stores {
            if dst_to_local.contains_key(&dst) {
                continue;
            }
            let local = self
                .insert_after(last_local, Op::Define { dtype: self.dtype(dst), scope: MemScope::Local, ro: false, len: 1024 });
            last_local = local;
            dst_to_local.insert(dst, local);
        }

        // Step 11: Replace each global store with a store to local at combined_idx
        let mut processed_dst: Set<OpId> = Set::default();
        for &(store_op, dst, val, _) in &global_stores {
            let local = dst_to_local[&dst];
            self.ops[store_op].op =
                Op::Store { dst: local, x: val, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        // Step 12: Insert barrier after the last store, then scalar loads + global stores
        let barrier = self.insert_after(global_stores.last().unwrap().0, Op::Barrier);
        let mut insert_point = barrier;
        for &(_, dst, _, store_idx) in &global_stores {
            if !processed_dst.insert(dst) {
                continue;
            }
            let local = dst_to_local[&dst];
            let scalar_load =
                self.insert_after(insert_point, Op::Load { src: local, index: combined_idx, layout: MemLayout::Scalar });
            insert_point = scalar_load;
            let global_store =
                self.insert_after(insert_point, Op::Store { dst, x: scalar_load, index: store_idx, layout: MemLayout::Scalar });
            insert_point = global_store;
        }

        self.verify();
    }

    fn tenstorrent_group(&mut self) {
        let mut barriers = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Barrier = self.at(op_id) {
                barriers.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if barriers.len() != 2 {
            return;
        }
        let barrier1 = barriers[0];
        let barrier2 = barriers[1];

        let mut stores1 = Vec::new();
        let mut stores2 = Vec::new();
        let mut stores3 = Vec::new();
        let mut phase = 0u8;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if op_id == barrier1 {
                phase = 1;
            } else if op_id == barrier2 {
                phase = 2;
            } else if let Op::Store { .. } = self.at(op_id) {
                match phase {
                    0 => stores1.push(op_id),
                    1 => stores2.push(op_id),
                    2 => stores3.push(op_id),
                    _ => unreachable!(),
                }
            }
            op_id = self.next_op(op_id);
        }

        let set1 = gather_deps(self, &stores1);
        let set2 = gather_deps(self, &stores2);

        let is_sticky = |k: &Kernel, id: OpId| -> bool { matches!(k.ops[id].op, Op::Define { .. } | Op::Const(_) | Op::Barrier) };

        let mut order_rev = Vec::new();
        let mut op_id = self.tail;
        while !op_id.is_null() {
            order_rev.push(op_id);
            op_id = self.prev_op(op_id);
        }

        for op_id in order_rev {
            if is_sticky(self, op_id) {
                continue;
            }
            if !set1.contains(&op_id) && !set2.contains(&op_id) {
                self.move_op_after(op_id, barrier2);
            } else if !set1.contains(&op_id) {
                self.move_op_after(op_id, barrier1);
            }
        }

        self.verify();
    }

    fn tenstorrent_loop_local(&mut self) {
        let mut lidxs: Vec<(u32, OpId, u32)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, axis, scope: IdxScope::Local } = self.at(op_id) {
                lidxs.push((*axis, op_id, *len as u32));
            }
            op_id = self.next_op(op_id);
        }
        if lidxs.is_empty() {
            return;
        }
        lidxs.sort_by_key(|&(axis, _, _)| axis);

        let mut barriers = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Barrier = self.at(op_id) {
                barriers.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if barriers.len() != 2 {
            return;
        }
        let barrier1 = barriers[0];
        let barrier2 = barriers[1];

        let first_lidx = lidxs.first().unwrap().1;
        let const_32 = self.insert_before(first_lidx, Op::Const(Constant::idx(32u32)));

        for &(_axis, id, _len) in &lidxs {
            self.ops[id].op = Op::Loop { len: const_32 };
        }

        for &(_axis, _id, _len) in lidxs.iter().rev() {
            self.insert_before(barrier1, Op::EndLoop);
        }

        let mut new_loops = Vec::new();
        let mut insert_after = barrier2;
        for &(_axis, _id, _len) in &lidxs {
            let loop_op = self.insert_after(insert_after, Op::Loop { len: const_32 });
            new_loops.push(loop_op);
            insert_after = loop_op;
        }

        let mut replace_map: Map<OpId, OpId> = Map::default();
        for ((_axis, old_id, _len), new_loop) in lidxs.iter().zip(new_loops.iter()) {
            replace_map.insert(*old_id, *new_loop);
        }

        let mut op_id = self.next_op(barrier2);
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters_mut() {
                if let Some(&new_id) = replace_map.get(param) {
                    *param = new_id;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Clone dependency chain inside write-back loops so per-thread index
        // computation (which depends on local indices) is re-computed using
        // the new loop variables instead of stale values from the preload section.
        let inner_loop = new_loops.last().unwrap();
        let first_inside = self.next_op(*inner_loop);
        let mut write_back_ops = Vec::new();
        let mut op_id = self.next_op(barrier2);
        while !op_id.is_null() {
            write_back_ops.push(op_id);
            op_id = self.next_op(op_id);
        }
        let deps = gather_deps(self, &write_back_ops);
        let mut clone_map = replace_map.clone();
        let mut deps_sorted: Vec<OpId> = deps.iter().copied().collect();
        deps_sorted.sort_by_key(|&id| {
            let mut order = 0u32;
            let mut scan = self.head;
            while !scan.is_null() && scan != id {
                order += 1;
                scan = self.next_op(scan);
            }
            order
        });
        let mut insert_point = *inner_loop;
        for &dep_id in &deps_sorted {
            let mut inside = false;
            let mut scan = first_inside;
            while !scan.is_null() {
                if scan == dep_id {
                    inside = true;
                    break;
                }
                scan = self.next_op(scan);
            }
            if inside {
                continue;
            }
            if matches!(self.at(dep_id), Op::Define { .. } | Op::Const(_) | Op::Index { .. } | Op::Barrier | Op::Loop { .. }) {
                continue;
            }
            let mut cloned_op = self.at(dep_id).clone();
            for param in cloned_op.parameters_mut() {
                if let Some(&new_id) = clone_map.get(param) {
                    *param = new_id;
                }
            }
            let new_id = self.insert_after(insert_point, cloned_op);
            insert_point = new_id;
            clone_map.insert(dep_id, new_id);
        }
        let mut op_id = first_inside;
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters_mut() {
                if let Some(&new_id) = clone_map.get(param) {
                    *param = new_id;
                }
            }
            op_id = self.next_op(op_id);
        }

        for &(_axis, _id, _len) in lidxs.iter().rev() {
            self.push_back(Op::EndLoop);
        }

        self.loop_invariant_code_motion();

        self.verify();
    }
}
