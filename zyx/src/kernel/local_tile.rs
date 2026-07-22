// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

const TILE_NELT: Dim = 1024; // TT tiles are 32×32 = 1024 elements
const TILE_T: MemLayout = MemLayout::Tile { x: 32, y: 32, stride: 32 };

impl Kernel {
    /// Tile 1D global indices: divide gidx by tile size and wrap access in a loop.
    ///
    /// Restructures the kernel into three phases:
    /// 1. Scalar global→local loop (reader)  — copy elements into local tile buffer
    /// 2. Tile compute                       — load tiles, compute, store tiles
    /// 3. Scalar local→global loop (writer)  — copy results back
    pub(crate) fn tenstorrent_tile(&mut self) {
        let Some((gidx, orig_len)) = self.find_gidx() else {
            return;
        };

        // Divide gidx length by tile size to make it a tile index
        let n_tiles = (orig_len + TILE_NELT - 1) / TILE_NELT;
        if let Op::Index { len, .. } = &mut self.ops[gidx].op {
            *len = n_tiles;
        }

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

        // Find the last global define to insert local defines after it
        let mut last_global = self.head;
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Define { scope: Scope::Global, .. }) {
                last_global = scan;
            }
            scan = self.next_op(scan);
        }

        // Allocate local buffers for inputs: trace through cast chain to find compute dtype.
        let mut insert_point = last_global;
        let mut in_locals: Map<OpId, OpId> = Map::default();
        for &(lid, _src) in &load_ids {
            let dt = self.resolve_compute_dtype(lid);
            let local = self.insert_after(insert_point, Op::Define { dtype: dt, scope: Scope::Local, ro: false, len: TILE_NELT });
            in_locals.insert(lid, local);
            insert_point = local;
        }

        // Allocate local buffers for outputs: one per store, typed as store value dtype
        let mut out_locals: Map<OpId, OpId> = Map::default();
        for &(sid, _dst) in &store_ids {
            let x = match self.at(sid) {
                Op::Store { x, .. } => *x,
                _ => unreachable!(),
            };
            let dt = self.dtype(x);
            let local = self.insert_after(insert_point, Op::Define { dtype: dt, scope: Scope::Local, ro: false, len: TILE_NELT });
            out_locals.insert(sid, local);
            insert_point = local;
        }

        // ── Phase 1: Scalar global → local loop ──
        // for i in 0..TILE_NELT:
        //     local[i] = global[gidx * TILE_NELT + i]
        let first_load = load_ids[0].0;
        let loop_p1 = self.insert_after(first_load, Op::Loop { len: orig_len });
        let const_nelt = self.insert_after(loop_p1, Op::Const(Constant::idx(TILE_NELT as u64)));
        let scaled = self.insert_after(const_nelt, Op::Binary { x: gidx, y: const_nelt, bop: BOp::Mul });
        let elem_idx = self.insert_after(scaled, Op::Binary { x: scaled, y: loop_p1, bop: BOp::Add });
        let mut loop_end = elem_idx;
        for &(lid, _src) in &load_ids {
            let local = in_locals[&lid];
            let dup = self.insert_after(
                loop_end,
                Op::Load {
                    src: match self.at(lid) {
                        Op::Load { src, .. } => *src,
                        _ => unreachable!(),
                    },
                    index: elem_idx,
                    layout: MemLayout::Scalar,
                },
            );
            loop_end = self.insert_after(dup, Op::Store { dst: local, x: dup, index: loop_p1, layout: MemLayout::Scalar });
        }
        self.insert_after(loop_end, Op::EndLoop);

        // ── Phase 2: Tile compute (inserted before each original global store) ──
        let first_sid = store_ids[0].0;

        // Barrier separating phase 1 and phase 2
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

            // ── Phase 3: Scalar local → global loop ──
            // for i in 0..TILE_NELT:
            //     global[gidx * TILE_NELT + i] = local[i]
            let loop_p3 = self.insert_before(sid, Op::Loop { len: orig_len });
            let const_nelt3 = self.insert_after(loop_p3, Op::Const(Constant::idx(TILE_NELT as u64)));
            let scaled3 = self.insert_after(const_nelt3, Op::Binary { x: gidx, y: const_nelt3, bop: BOp::Mul });
            let elem_idx3 = self.insert_after(scaled3, Op::Binary { x: scaled3, y: loop_p3, bop: BOp::Add });
            let ll = self.insert_after(elem_idx3, Op::Load { src: out_local, index: loop_p3, layout: MemLayout::Scalar });
            let store_p3 = self.insert_after(ll, Op::Store { dst: _dst, x: ll, index: elem_idx3, layout: MemLayout::Scalar });
            self.insert_after(store_p3, Op::EndLoop);

            // Remove the original store (replaced by phase 3)
            self.remove_op(sid);
        }
    }

    // ── helpers ──

    fn find_gidx(&self) -> Option<(OpId, Dim)> {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope: Scope::Global, axis: 0 } = self.at(op_id) {
                if *len < TILE_NELT && *len > 0 {
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

    /// Trace forward from a load through cast ops to find the compute dtype.
    fn resolve_compute_dtype(&self, load_id: OpId) -> DType {
        let mut cur = self.next_op(load_id);
        let mut prev = load_id;
        while !cur.is_null() {
            match self.at(cur) {
                Op::Cast { x, dtype } if *x == prev => {
                    prev = cur;
                    cur = self.next_op(cur);
                }
                _ => break,
            }
        }
        self.dtype(prev)
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
