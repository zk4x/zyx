// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemLayout, MemScope, Op, OpId},
    shape::Dim,
    slab::SlabId,
};

fn gather_deps(kernel: &Kernel, seeds: &[OpId]) -> Set<OpId> {
    let mut visited = Set::default();
    let mut stack: Vec<OpId> = seeds.to_vec();
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
            if self.ops.values().filter(|node| matches!(node.op, Op::Loop { .. })).count() == 1 {
                self.tenstorrent_scalar_reduce_single_loop();
            } else {
                // TODO row reduce, matmul, transpose tile
                todo!();
            }
        } else {
            self.tenstorrent_elementwise();
        }
    }

    fn tenstorrent_scalar_reduce_single_loop(&mut self) {
        self.tenstorrent_pad_loops();
        self.tenstorrent_single_scalar_tile();
    }

    fn tenstorrent_elementwise(&mut self) {
        self.tenstorrent_pad_gidx();
        self.tenstorrent_local();
        self.tenstorrent_group();
        self.tenstorrent_loop_local();
    }

    fn tenstorrent_pad_gidx(&mut self) {
        let mut gidxs: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { axis, kind } = *self.at(op_id) {
                if let IdxKind::Group(len) = kind {
                    gidxs.push((op_id, axis, self.resolve_dim(len).unwrap()));
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
                let new_len = if let Op::Index { kind: IdxKind::Group(len), .. } = self.ops[id].op {
                    self.resolve_dim(len).unwrap()
                } else {
                    unreachable!()
                };
                let sqrt = (new_len as f64).sqrt() as Dim;
                let f1 = (32..=sqrt).rev().find(|&f| f % 32 == 0 && new_len % f == 0);
                let Some(f1) = f1 else {
                    return;
                };
                let f2 = new_len / f1;
                let f1_id = self.const_idx(f1);
                let f2_id = self.const_idx(f2);
                self.split_dim(
                    id,
                    vec![
                        Op::Index { axis: 0, kind: IdxKind::Group(f1_id) },
                        Op::Index { axis: 1, kind: IdxKind::Group(f2_id) },
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

    /// Pad all loops to length 1024.
    fn tenstorrent_pad_loops(&mut self) {
        let loops: Vec<OpId> = self.ops.iter().filter(|(_, node)| matches!(node.op, Op::Loop { .. })).map(|(id, _)| id).collect();
        for loop_id in loops {
            let Op::Loop { len: len_id } = self.ops[loop_id].op else {
                continue;
            };
            let len = self.resolve_dim(len_id).unwrap();
            let pad = round_up(len, 1024);
            if pad > 0 {
                self.pad_loop(loop_id, pad);
            }
        }
        self.verify();
    }

    fn tenstorrent_local(&mut self) {
        // Split each GroupIndex into GroupIndex(len/32) + Local(32)
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { axis, kind: IdxKind::Group(len) } = self.ops[op_id].op {
                let len = self.resolve_dim(len).unwrap();
                if len.is_multiple_of(32) && len >= 32 {
                    let f1 = len / 32;
                    let f1_id = self.const_idx(f1);
                    self.split_dim(
                        op_id,
                        vec![
                            Op::Index { axis, kind: IdxKind::Group(f1_id) },
                            Op::Index { axis, kind: IdxKind::Local(32) },
                        ],
                    );
                } else {
                    return;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Verify exactly 2 local indices of len 32 (axes 0 and 1)
        let mut lidxs = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { axis, kind: IdxKind::Local(len) } = self.ops[op_id].op {
                if len != 32 {
                    return;
                }
                lidxs.push((axis, op_id));
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

        // Find all scalar loads from global params
        let global_loads: Vec<(OpId, OpId, OpId)> = {
            let mut loads = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Load { src, index, layout: MemLayout::Scalar } = self.at(op_id)
                    && matches!(self.at(*src), Op::Storage { scope: MemScope::Global, .. })
                {
                    loads.push((op_id, *src, *index));
                }
                op_id = self.next_op(op_id);
            }
            loads
        };

        if global_loads.is_empty() {
            return;
        }

        // Insert constant/index computation before the first load
        let first_load = global_loads[0].0;
        let first_global_idx = global_loads[0].2;
        let const_32 = self.insert_before(first_load, Op::Const(Constant::idx(32u32)));
        let scaled = self.insert_before(first_load, Op::Binary { x: lidx0, y: const_32, bop: BOp::Mul });
        let combined_idx = self.insert_before(first_load, Op::Binary { x: scaled, y: lidx1, bop: BOp::Add });
        let zero = self.insert_before(first_load, Op::Const(Constant::idx(0u32)));

        // Find the last global param to insert locals after it
        let mut last_global = self.head;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Storage { scope: MemScope::Global, .. }) {
                last_global = scan;
            }
            scan = self.next_op(scan);
        }

        // Allocate local buffers for each unique global source
        let mut src_to_local: Map<OpId, OpId> = Map::default();
        for &(_, src, _) in &global_loads {
            if src_to_local.contains_key(&src) {
                continue;
            }
            let local = self.insert_after(
                last_global,
                Op::Storage { dtype: self.dtype(src), scope: MemScope::Circular, len: 1024 },
            );
            last_global = local;
            src_to_local.insert(src, local);
        }

        // Insert all global→local stores before the first load, then a barrier.
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
                    Op::Store { dst: local, src: global_load, index: combined_idx, layout: MemLayout::Scalar },
                );
            }
        }
        self.insert_before(first_load, Op::Barrier);

        // Replace all original loads with tiled loads from local
        for &(load_op, src, _) in &global_loads {
            let local = src_to_local[&src];
            self.ops[load_op].op = Op::Load { src: local, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        // Find all scalar stores to global params
        let global_stores: Vec<(OpId, OpId, OpId, OpId)> = {
            let mut stores = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Store { dst, src: x, index, layout: MemLayout::Scalar } = self.at(op_id)
                    && matches!(self.at(*dst), Op::Storage { scope: MemScope::Global, .. })
                {
                    stores.push((op_id, *dst, *x, *index));
                }
                op_id = self.next_op(op_id);
            }
            stores
        };

        if global_stores.is_empty() {
            return;
        }

        // Allocate local buffers for each unique global destination
        let mut dst_to_local: Map<OpId, OpId> = Map::default();
        let mut last_local = last_global;
        for &(_, dst, _, _) in &global_stores {
            if dst_to_local.contains_key(&dst) {
                continue;
            }
            let local = self.insert_after(
                last_local,
                Op::Storage { dtype: self.dtype(dst), scope: MemScope::Circular, len: 1024 },
            );
            last_local = local;
            dst_to_local.insert(dst, local);
        }

        // Replace each global store with a store to local at combined_idx
        let mut processed_dst: Set<OpId> = Set::default();
        for &(store_op, dst, val, _) in &global_stores {
            let local = dst_to_local[&dst];
            self.ops[store_op].op =
                Op::Store { dst: local, src: val, index: zero, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        // Insert barrier after the last store, then scalar loads + global stores
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
                self.insert_after(insert_point, Op::Store { dst, src: scalar_load, index: store_idx, layout: MemLayout::Scalar });
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

        let is_sticky = |k: &Kernel, id: OpId| -> bool { matches!(k.ops[id].op, Op::Storage { .. } | Op::Const(_) | Op::Barrier) };

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
            if let Op::Index { axis, kind: IdxKind::Local(len) } = self.ops[op_id].op {
                lidxs.push((axis, op_id, len));
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
            if matches!(self.at(dep_id), Op::Storage { .. } | Op::Const(_) | Op::Index { .. } | Op::Barrier | Op::Loop { .. }) {
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

    /// Step 1: reader kernel for single-scalar-tile reduction.
    ///
    /// This pass transforms a kernel that reduces a global buffer to a single
    /// scalar by gathering global scalar loads inside the loop, pre-loading
    /// them into circular buffers using a tiled two-loop structure, and
    /// replacing the original global loads with loads from those buffers.
    ///
    /// The transformation works as follows:
    /// 1. Find the single loop in the kernel and all global scalar loads
    ///    inside its body. Assert that no global loads exist past the loop.
    /// 2. For each unique global source, create a circular buffer storage (len=1024)
    ///    right after the last global param.
    /// 3. Insert two nested loops of length 32. The original loop index is
    ///    reconstructed as `mad(outer_loop, 32, inner_loop)`.
    /// 4. Replay each load's indexing chain, replacing the original loop id
    ///    with the mad result, and use the rebuilt index to load globals into
    ///    the circular buffers.
    /// 5. Close both loops with EndLoop.
    ///
    /// The original loop and all computation ops remain untouched; this pass
    /// only inserts new ops after the last global param.
    ///
    /// Step 2: compute kernel.
    ///
    /// Transforms the kernel to use circular buffers and a tiled accumulator:
    /// 1. Asserts no global loads exist before the loop.
    /// 2. Divides the loop length constant by 1024 (loop now runs once).
    /// 3. Redirects global loads to load from the circular buffers instead.
    /// 4. Converts the scalar accumulator register to a tile of 1024 elements.
    /// 5. Finds the BOp from the accumulation inside the loop.
    /// 6. Inserts a ReduceTile (scalar) after the loop to reduce the tile.
    /// 7. Remaps all uses of the accumulator register after the loop to the ReduceTile result.
    fn tenstorrent_single_scalar_tile(&mut self) {
        self.debug();

        // Find the single loop in the kernel
        let loop_id = {
            let mut op_id = self.head;
            let mut found = OpId::NULL;
            while !op_id.is_null() {
                if matches!(self.at(op_id), Op::Loop { .. }) {
                    found = op_id;
                    break;
                }
                op_id = self.next_op(op_id);
            }
            found
        };
        debug_assert!(!loop_id.is_null());

        // Debug assert no global loads before the loop
        let mut op_id = self.head;
        while op_id != loop_id {
            debug_assert!(
                !matches!(
                    self.at(op_id),
                    Op::Load { src, .. } if matches!(self.at(*src), Op::Storage { scope: MemScope::Global, .. })
                ),
                "global loads should have been moved into circular buffers"
            );
            op_id = self.next_op(op_id);
        }

        // Find matching EndLoop
        let endloop_id = {
            let mut depth = 0;
            let mut op_id = loop_id;
            loop {
                match self.at(op_id) {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => {
                        depth -= 1;
                        if depth == 0 {
                            break op_id;
                        }
                    }
                    _ => {}
                }
                op_id = self.next_op(op_id);
            }
        };

        // Gather all global scalar loads inside the loop body
        let mut global_loads = Vec::new();
        let mut op_id = self.next_op(loop_id);
        while op_id != endloop_id {
            if let Op::Load { src, index, layout: MemLayout::Scalar } = self.at(op_id)
                && matches!(self.at(*src), Op::Storage { scope: MemScope::Global, .. })
            {
                global_loads.push((op_id, *src, *index));
            }
            op_id = self.next_op(op_id);
        }

        if global_loads.is_empty() {
            self.debug();
            return;
        }

        // Debug assert no global loads past the loop
        let mut op_id = self.next_op(endloop_id);
        while !op_id.is_null() {
            if let Op::Load { src, .. } = self.at(op_id) {
                debug_assert!(!matches!(self.at(*src), Op::Storage { scope: MemScope::Global, .. }));
            }
            op_id = self.next_op(op_id);
        }

        // Find last global param
        let mut last_global = self.head;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Storage { scope: MemScope::Global, .. }) {
                last_global = scan;
            }
            scan = self.next_op(scan);
        }

        // Create circular buffer storages for each unique global source
        let mut src_to_cb: Map<OpId, OpId> = Map::default();
        for &(_, src, _) in &global_loads {
            if src_to_cb.contains_key(&src) {
                continue;
            }
            let cb = self.insert_after(
                last_global,
                Op::Storage { dtype: self.dtype(src), scope: MemScope::Circular, len: 1024 },
            );
            last_global = cb;
            src_to_cb.insert(src, cb);
        }

        // Insert two loops, each length 32
        let const_32 = self.insert_after(last_global, Op::Const(Constant::idx(32u32)));
        let outer_loop = self.insert_after(const_32, Op::Loop { len: const_32 });
        let const_32_inner = self.insert_after(outer_loop, Op::Const(Constant::idx(32u32)));
        let inner_loop = self.insert_after(const_32_inner, Op::Loop { len: const_32_inner });

        // Create mad(outer_loop, 32, inner_loop) representing the original loop index
        let mad_id = self.insert_after(inner_loop, Op::Mad { x: outer_loop, y: const_32, z: inner_loop });

        // Replay the indexing chain for each global load, replacing the original
        // loop id with mad_id, then load globals into circular buffers.
        let mut clone_map: Map<OpId, OpId> = Map::default();
        clone_map.insert(loop_id, mad_id);

        let mut insert_point = mad_id;
        for &(_, src, index) in &global_loads {
            let cb = src_to_cb[&src];

            // Gather transitive deps of this load's index
            let deps = gather_deps(self, &[index]);
            let mut sorted_deps: Vec<OpId> = deps.iter().copied().collect();
            sorted_deps.sort_by_key(|&id| {
                let mut order = 0u32;
                let mut scan = self.head;
                while !scan.is_null() && scan != id {
                    order += 1;
                    scan = self.next_op(scan);
                }
                order
            });

            // Clone non-sticky ops, replacing loop id with mad_id
            for &dep_id in &sorted_deps {
                if dep_id == loop_id {
                    continue;
                }
                if matches!(
                    self.at(dep_id),
                    Op::Storage { .. } | Op::Const(_) | Op::Index { .. } | Op::Barrier | Op::Loop { .. } | Op::EndLoop
                ) {
                    clone_map.entry(dep_id).or_insert(dep_id);
                    continue;
                }
                if clone_map.contains_key(&dep_id) {
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

            let rebuilt_index = clone_map[&index];

            // Load from global into circular buffer using rebuilt index
            let load_id = self.insert_after(insert_point, Op::Load { src, index: rebuilt_index, layout: MemLayout::Scalar });
            insert_point = load_id;
        }

        // Close the two loops
        self.insert_after(insert_point, Op::EndLoop);
        self.insert_after(insert_point, Op::EndLoop);

        // Step 2: compute kernel

        // Divide the loop length constant by 1024
        let loop_len_id = match self.at(loop_id) {
            Op::Loop { len } => *len,
            _ => unreachable!(),
        };
        if let Op::Const(c) = &self.ops[loop_len_id].op {
            let new_val = match c {
                Constant::U32(v) => Constant::U32(v / 1024),
                Constant::U64(v) => {
                    let val = u64::from_le_bytes(*v);
                    Constant::U64((val / 1024).to_le_bytes())
                }
                _ => unreachable!(),
            };
            self.ops[loop_len_id].op = Op::Const(new_val);
        }

        // Edit global loads to load from circular buffers using tiled layout
        for &(load_op, src, _) in &global_loads {
            let cb = src_to_cb[&src];
            if let Op::Load { index, .. } = self.at(load_op).clone() {
                self.ops[load_op].op = Op::Load { src: cb, index, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
            }
        }

        // Find the accumulator register storage inside the loop
        let mut accumulator = OpId::NULL;
        let mut op_id = self.next_op(loop_id);
        while op_id != endloop_id {
            if let Op::Store { dst, .. } = self.at(op_id)
                && matches!(self.at(*dst), Op::Storage { scope: MemScope::Register, .. })
            {
                accumulator = *dst;
                break;
            }
            op_id = self.next_op(op_id);
        }
        debug_assert!(!accumulator.is_null());

        // Convert accumulator to register tile, length 1024
        if let Op::Storage { len, .. } = &mut self.ops[accumulator].op {
            *len = 1024;
        }

        // Change accumulator store and load to tiled layout
        let mut op_id = self.next_op(loop_id);
        while op_id != endloop_id {
            match self.at(op_id) {
                Op::Store { dst, src: x, index, layout } if *dst == accumulator => {
                    self.ops[op_id].op =
                        Op::Store { dst: *dst, src: *x, index: *index, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
                }
                Op::Load { src, index, layout } if *src == accumulator => {
                    self.ops[op_id].op =
                        Op::Load { src: *src, index: *index, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Find the BOp used in the accumulation inside the loop
        let mut accumulation_bop = BOp::Add;
        let mut op_id = self.next_op(loop_id);
        while op_id != endloop_id {
            if let Op::Store { dst, src: x, .. } = self.at(op_id)
                && *dst == accumulator
            {
                if let Op::Binary { bop, .. } = self.at(*x) {
                    accumulation_bop = *bop;
                }
                break;
            }
            op_id = self.next_op(op_id);
        }

        // Add ReduceTile after the loop
        let reduce_tile = self.insert_after(
            endloop_id,
            Op::ReduceTile { x: accumulator, rop: accumulation_bop, kind: crate::kernel::ops::TileReduceKind::Scalar },
        );

        // Remap all uses of the accumulator after the loop to the ReduceTile result
        let mut op_id = self.next_op(reduce_tile);
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters_mut() {
                if *param == accumulator {
                    *param = reduce_tile;
                }
            }
            op_id = self.next_op(op_id);
        }

        self.debug();

        todo!();
    }
}
