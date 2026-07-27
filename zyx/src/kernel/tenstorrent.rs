// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

fn round_up(len: Dim, multiple: Dim) -> Dim {
    let rem = len % multiple;
    if rem == 0 { 0 } else { multiple - rem }
}

impl Kernel {
    pub(crate) fn opt_tenstorrent_pad(&mut self) {
        let mut gidxs: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::GroupIndex { len, axis } = self.at(op_id) {
                gidxs.push((op_id, axis, len));
            }
            // Can't run this optimization on kernel that already has local indices
            if let Op::LocalIndex { .. } = self.at(op_id) {
                return;
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
                let new_len = if let Op::GroupIndex { len, .. } = self.at(id) {
                    *len
                } else {
                    unreachable!()
                };
                let f1 = (new_len as f64).sqrt() as Dim;
                let f1 = (2..=f1).rev().find(|&f| new_len % f == 0).unwrap_or(1);
                if f1 <= 1 || f1 == new_len {
                    return;
                }
                let f2 = new_len / f1;
                self.split_dim(id, vec![Op::GroupIndex { len: f1, axis: 0 }, Op::GroupIndex { len: f2, axis: 1 }]);
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
    }

    pub(crate) fn opt_tenstorrent_local(&mut self) {
        // Step 1: Split each GroupIndex into GroupIndex(len/32) + Loop(32)
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::GroupIndex { len, axis } = self.ops[op_id].op {
                if len % 32 == 0 && len >= 32 {
                    let f1 = len / 32;
                    self.split_dim(op_id, vec![Op::GroupIndex { len: f1, axis }, Op::LocalIndex { len, axis }]);
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
            if let Op::LocalIndex { len, axis } = self.at(op_id) {
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
        let (lidx0, lidx1) = (lidxs[0].1, lidxs[1].1);

        // Step 3: Find all scalar loads from global defines
        let global_loads: Vec<(OpId, OpId)> = {
            let mut loads = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Load { src, layout: MemLayout::Scalar, .. } = self.at(op_id) {
                    if matches!(self.at(*src), Op::Define { scope: Scope::Global, .. }) {
                        loads.push((op_id, *src));
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
        let const_32 = self.insert_before(first_load, Op::Const(Constant::idx(32u32)));
        let scaled = self.insert_before(first_load, Op::Binary { x: lidx0, y: const_32, bop: BOp::Mul });
        let combined_idx = self.insert_before(first_load, Op::Binary { x: scaled, y: lidx1, bop: BOp::Add });
        let zero = self.insert_before(first_load, Op::Const(Constant::idx(0u32)));

        // Step 5: Find the last global define to insert locals after it
        let mut last_global = self.head;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Define { scope: Scope::Global, .. }) {
                last_global = scan;
            }
            scan = self.next_op(scan);
        }

        // Step 6: Allocate local buffers for each unique global source
        let mut src_to_local: Map<OpId, OpId> = Map::default();
        for &(_, src) in &global_loads {
            if src_to_local.contains_key(&src) {
                continue;
            }
            let local =
                self.insert_after(last_global, Op::Define { dtype: self.dtype(src), scope: Scope::Local, ro: false, len: 1024 });
            last_global = local;
            src_to_local.insert(src, local);
        }

        // Step 7: Insert all global→local stores before the first load, then a barrier
        let mut processed: Set<OpId> = Set::default();
        for &(_, src) in &global_loads {
            if processed.insert(src) {
                let local = src_to_local[&src];
                let global_load = self.insert_before(first_load, Op::Load { src, index: combined_idx, layout: MemLayout::Scalar });
                self.insert_before(first_load, Op::Store { dst: local, x: global_load, index: combined_idx, layout: MemLayout::Scalar });
            }
        }
        self.insert_before(first_load, Op::Barrier);

        // Step 8: Replace all original loads with tiled loads from local
        for &(load_op, src) in &global_loads {
            let local = src_to_local[&src];
            self.ops[load_op].op = Op::Load { src: local, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        // Step 9: Find all scalar stores to global defines
        let global_stores: Vec<(OpId, OpId, OpId, OpId)> = {
            let mut stores = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Store { dst, x, index, layout: MemLayout::Scalar } = self.at(op_id) {
                    if matches!(self.at(*dst), Op::Define { scope: Scope::Global, .. }) {
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
            let local =
                self.insert_after(last_local, Op::Define { dtype: self.dtype(dst), scope: Scope::Local, ro: false, len: 1024 });
            last_local = local;
            dst_to_local.insert(dst, local);
        }

        // Step 11: Replace each global store with a store to local at combined_idx
        let mut processed_dst: Set<OpId> = Set::default();
        for &(store_op, dst, val, _) in &global_stores {
            let local = dst_to_local[&dst];
            self.ops[store_op].op = Op::Store { dst: local, x: val, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
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
                self.insert_after(insert_point, Op::Load { src: local, index: store_idx, layout: MemLayout::Scalar });
            insert_point = scalar_load;
            let global_store = self.insert_after(insert_point, Op::Store { dst, x: scalar_load, index: store_idx, layout: MemLayout::Scalar });
            insert_point = global_store;
        }
    }
}
