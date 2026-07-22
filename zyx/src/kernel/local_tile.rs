// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

const TILE_WIDTH: Dim = 32;
const TILE_T: MemLayout = MemLayout::Tile { x: 32, y: 1, stride: 32 };

impl Kernel {
    /// Create local memory tiles for small 1D global indices.
    ///
    /// Restructures the kernel into three phases:
    /// 1. Scalar global→local (reader)  — store global data to local buffers
    /// 2. Tile compute                  — load tiles, compute, store tiles
    /// 3. Scalar local→global (writer)  — load scalar from local, store to global
    pub(crate) fn tile_local(&mut self) {
        let Some((gidx, _orig_len)) = self.find_small_gidx() else {
            return;
        };
        let padded = round_up(_orig_len, TILE_WIDTH);

        // Collect global scalar loads and stores using this gidx
        let mut load_ids: Vec<(OpId, OpId)> = Vec::new(); // (load_op, src_def)
        let mut store_ids: Vec<(OpId, OpId)> = Vec::new(); // (store_op, dst_def)
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Load { src, index, layout: MemLayout::Scalar } => {
                    if self.is_global_def(*src) && self.depends_on(*index, gidx, &mut Set::default()) {
                        load_ids.push((op_id, *src));
                    }
                }
                Op::Store { dst, x: _, index, layout: MemLayout::Scalar } => {
                    if self.is_global_def(*dst) && self.depends_on(*index, gidx, &mut Set::default()) {
                        store_ids.push((op_id, *dst));
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        if load_ids.is_empty() || store_ids.is_empty() {
            return;
        }

        // Allocate local buffers for inputs: one per load, typed as load's result dtype
        // Insert at head so defines precede all references (dtypes computation walks sequentially).
        let head = self.head;
        let mut in_locals: Map<OpId, OpId> = Map::default();
        for &(lid, _src) in &load_ids {
            let dt = self.dtype(lid);
            let local = self.insert_before(head, Op::Define { dtype: dt, scope: Scope::Local, ro: false, len: padded });
            in_locals.insert(lid, local);
        }

        // Allocate local buffers for outputs: one per store, typed as store value dtype
        let mut out_locals: Map<OpId, OpId> = Map::default();
        for &(sid, _dst) in &store_ids {
            let x = match self.at(sid) {
                Op::Store { x, .. } => *x,
                _ => unreachable!(),
            };
            let dt = self.dtype(x);
            let local = self.insert_before(head, Op::Define { dtype: dt, scope: Scope::Local, ro: false, len: padded });
            out_locals.insert(sid, local);
        }

        // ── Phase 1: Scalar global → local ──
        for &(lid, _src) in &load_ids {
            let local = in_locals[&lid];
            // Use a duplicate scalar load so the phase 1 store does NOT depend on `lid`
            // (which will be remapped to a tile load in phase 2).
            let dup = self.insert_after(
                lid,
                Op::Load {
                    src: match self.at(lid) {
                        Op::Load { src, .. } => *src,
                        _ => unreachable!(),
                    },
                    index: gidx,
                    layout: MemLayout::Scalar,
                },
            );
            self.insert_after(dup, Op::Store { dst: local, x: dup, index: gidx, layout: MemLayout::Scalar });
        }

        // ── Phase 2: Tile compute (inserted before each original global store) ──
        let first_sid = store_ids[0].0;

        // Barrier separating phase 1 and phase 2 (insert before first store = leftmost)
        self.insert_before(first_sid, Op::Barrier);

        // Tile loads (shared across all stores)
        let c0 = self.insert_before(first_sid, Op::Const(Constant::idx(0u32)));
        let mut tile_map: Map<OpId, OpId> = Map::default();
        for &(lid, _src) in &load_ids {
            let local = in_locals[&lid];
            let tl = self.insert_before(first_sid, Op::Load { src: local, index: c0, layout: TILE_T });
            tile_map.insert(lid, tl);
        }

        for &(sid, _dst) in &store_ids {
            let x = match self.at(sid) {
                Op::Store { x, .. } => *x,
                _ => unreachable!(),
            };
            let out_local = out_locals[&sid];
            // Fresh tile_map per store (only load mappings, not compute)
            let mut store_tile_map = tile_map.clone();
            let tile_result = self.clone_compute_tile(x, &mut store_tile_map, sid);
            // Don't need to merge store_tile_map back into tile_map
            self.insert_before(sid, Op::Store { dst: out_local, x: tile_result, index: c0, layout: TILE_T });

            // Barrier separating phase 2 and phase 3
            self.insert_before(sid, Op::Barrier);

            // ── Phase 3: Scalar local → global ──
            let ll = self.insert_before(sid, Op::Load { src: out_local, index: gidx, layout: MemLayout::Scalar });

            // Replace the original store with the phase 3 global store
            let target = &mut self.ops[sid].op;
            *target = Op::Store { dst: _dst, x: ll, index: gidx, layout: MemLayout::Scalar };
        }
    }

    // ── helpers ──

    fn find_small_gidx(&self) -> Option<(OpId, Dim)> {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope: Scope::Global, axis: 0 } = self.at(op_id) {
                if *len < TILE_WIDTH && *len > 0 {
                    return Some((op_id, *len));
                }
            }
            op_id = self.next_op(op_id);
        }
        None
    }

    fn is_global_def(&self, id: OpId) -> bool {
        matches!(self.at(id), Op::Define { scope: Scope::Global, .. })
    }

    /// Recursively clone a compute op and all its inputs as tile ops.
    /// All cloned ops are inserted before `before` using `insert_before`,
    /// which naturally produces left-to-right (dependency) order.
    fn clone_compute_tile(&mut self, op_id: OpId, tile_map: &mut Map<OpId, OpId>, before: OpId) -> OpId {
        if let Some(&tile) = tile_map.get(&op_id) {
            return tile;
        }
        let op = self.at(op_id).clone();
        let tile = match op {
            Op::Cast { x, dtype } => {
                let tile_x = self.clone_compute_tile(x, tile_map, before);
                self.insert_before(before, Op::Cast { x: tile_x, dtype })
            }
            Op::Unary { x, uop } => {
                let tile_x = self.clone_compute_tile(x, tile_map, before);
                self.insert_before(before, Op::Unary { x: tile_x, uop })
            }
            Op::Binary { x, y, bop } => {
                let tile_x = self.clone_compute_tile(x, tile_map, before);
                let tile_y = self.clone_compute_tile(y, tile_map, before);
                self.insert_before(before, Op::Binary { x: tile_x, y: tile_y, bop })
            }
            _ => op_id,
        };
        tile_map.insert(op_id, tile);
        tile
    }
}

fn round_up(n: Dim, multiple: Dim) -> Dim {
    (n + multiple - 1) / multiple * multiple
}
