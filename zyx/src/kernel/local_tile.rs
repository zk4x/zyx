// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Local tiling optimization.
//!
//! This module provides local tiling optimizations for kernels,
//! which tile local memory access patterns to improve performance.

#![allow(unused)]

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    fn max_global_load_batch(&self) -> usize {
        let mut max_batch = 0usize;
        let mut cur = 0usize;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match &self.ops[op_id].op {
                Op::Load { src, .. } => {
                    if matches!(
                        self.ops[*src].op,
                        Op::Define {
                            scope: Scope::Global,
                            ..
                        }
                    ) {
                        cur += 1;
                    } else {
                        max_batch = max_batch.max(cur);
                        cur = 0;
                    }
                }
                _ => {
                    max_batch = max_batch.max(cur);
                    cur = 0;
                }
            }
            op_id = self.next_op(op_id);
        }
        max_batch.max(cur)
    }

    /// Tile local memory access patterns.
    ///
    /// This method tiles local memory access patterns to improve
    /// performance by better utilizing local memory bandwidth.
    pub(crate) fn tile_local(&mut self) {
        // Find local indices (lidx) created by split_dim
        let mut lidxs: Vec<(OpId, Dim, u32)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index {
                len,
                scope: Scope::Local,
                axis,
            } = self.ops[op_id].op
            {
                lidxs.push((op_id, len, axis));
            }
            op_id = self.next_op(op_id);
        }
        if lidxs.is_empty() {
            eprintln!("=== tile_local: no lidx ops ===");
            return;
        }
        lidxs.sort_by_key(|(_, _, a)| *a);

        let max_batch = self.max_global_load_batch();
        eprintln!(
            "=== tile_local: max_batch={}, lidxs={:?} ===",
            max_batch,
            lidxs.iter().map(|(_, l, a)| (l, a)).collect::<Vec<_>>()
        );
        if max_batch == 0 {
            return;
        }

        // Find insertion point: first op after all structural ops
        let mut insert_point = self.head;
        while !insert_point.is_null() {
            match self.ops[insert_point].op {
                Op::Define { .. } | Op::Const(_) | Op::Index { .. } => {
                    insert_point = self.next_op(insert_point);
                }
                _ => break,
            }
        }
        if insert_point.is_null() {
            return;
        }

        // Build linear local index: lidx0 + lidx1 * dim0 + ...
        let tile_size: Dim = lidxs.iter().map(|(_, l, _)| *l).product();
        let mut lin_lidx = self.insert_before(insert_point, Op::Const(Constant::idx(0)));
        let mut stride: Dim = 1;
        for (lid, dim, _axis) in &lidxs {
            let st = self.insert_before(insert_point, Op::Const(Constant::idx(stride as u64)));
            let scaled = self.insert_before(
                insert_point,
                Op::Binary {
                    x: *lid,
                    y: st,
                    bop: BOp::Mul,
                },
            );
            lin_lidx = self.insert_before(
                insert_point,
                Op::Binary {
                    x: lin_lidx,
                    y: scaled,
                    bop: BOp::Add,
                },
            );
            stride *= dim;
        }

        // Determine dtype from first global load's source
        let tile_dtype = self
            .ops
            .iter()
            .find_map(|(_, node)| {
                if let Op::Load { src, .. } = node.op {
                    if let Op::Define {
                        dtype,
                        scope: Scope::Global,
                        ..
                    } = self.ops[src].op
                    {
                        return Some(dtype);
                    }
                }
                None
            })
            .unwrap_or(DType::F32);

        let tile_buf = self.insert_before(
            insert_point,
            Op::Define {
                dtype: tile_dtype,
                scope: Scope::Local,
                ro: false,
                len: tile_size * max_batch as Dim,
            },
        );

        // Walk all ops, batch consecutive global loads, wrap each batch
        let mut pending: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match &self.ops[op_id].op {
                Op::Load { src, .. } => {
                    if matches!(
                        self.ops[*src].op,
                        Op::Define {
                            scope: Scope::Global,
                            ..
                        }
                    ) {
                        pending.push(op_id);
                        op_id = next;
                        continue;
                    }
                }
                _ => {}
            }
            if !pending.is_empty() {
                self.flush_tile_batch(&pending, tile_buf, lin_lidx, tile_size);
                pending.clear();
            }
            op_id = next;
        }
        if !pending.is_empty() {
            self.flush_tile_batch(&pending, tile_buf, lin_lidx, tile_size);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    fn flush_tile_batch(&mut self, pending: &[OpId], tile_buf: OpId, lin_lidx: OpId, tile_size: Dim) {
        eprintln!(
            "=== flush: n={}, tile_buf={}, tile_size={} ===",
            pending.len(),
            tile_buf,
            tile_size
        );
        let n = pending.len();
        let mut insert_pt = pending[n - 1];

        // Create per-load position ops: pos_i = lin_lidx + i * tile_size
        let ts = self.insert_after(insert_pt, Op::Const(Constant::idx(tile_size)));
        insert_pt = ts;
        let mut positions: Vec<OpId> = Vec::with_capacity(n);
        for i in 0..n {
            let idx = self.insert_after(insert_pt, Op::Const(Constant::idx(i as u64)));
            insert_pt = idx;
            let scaled = self.insert_after(
                insert_pt,
                Op::Binary {
                    x: idx,
                    y: ts,
                    bop: BOp::Mul,
                },
            );
            insert_pt = scaled;
            let pos = self.insert_after(
                insert_pt,
                Op::Binary {
                    x: lin_lidx,
                    y: scaled,
                    bop: BOp::Add,
                },
            );
            insert_pt = pos;
            positions.push(pos);
        }

        // Insert Stores (each to its own position)
        for (&load_id, &pos) in pending.iter().zip(positions.iter()) {
            insert_pt = self.insert_after(
                insert_pt,
                Op::Store {
                    dst: tile_buf,
                    x: load_id,
                    index: pos,
                    layout: MemLayout::Scalar,
                },
            );
        }

        // One Barrier
        insert_pt = self.insert_after(insert_pt, Op::Barrier { scope: Scope::Local });

        // Insert Loads (each from its own position)
        let mut new_loads: Vec<OpId> = Vec::with_capacity(n);
        for &pos in positions.iter() {
            insert_pt = self.insert_after(
                insert_pt,
                Op::Load {
                    src: tile_buf,
                    index: pos,
                    layout: MemLayout::Scalar,
                },
            );
            new_loads.push(insert_pt);
        }

        // Remap consumers
        let mut remap = Map::default();
        for (&old_id, &new_id) in pending.iter().zip(new_loads.iter()) {
            remap.insert(old_id, new_id);
        }
        let mut walk = self.next_op(insert_pt);
        while !walk.is_null() {
            self.ops[walk].op.remap_params(&remap);
            walk = self.next_op(walk);
        }
    }
}
