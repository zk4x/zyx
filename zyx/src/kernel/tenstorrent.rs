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
        if let Op::GroupIndex { len, .. } = &mut self.ops[gidx].op {
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
        // Compute loop bound = min(orig_len - gidx * TILE_NELT, TILE_NELT)
        // = remaining + nelt - max(remaining, nelt)
        let first_load = load_ids[0].0;
        let p1_nelt = self.insert_after(first_load, Op::Const(Constant::idx(TILE_NELT as u64)));
        let p1_orig = self.insert_after(p1_nelt, Op::Const(Constant::idx(orig_len as u64)));
        let p1_scaled = self.insert_after(p1_orig, Op::Binary { x: gidx, y: p1_nelt, bop: BOp::Mul });
        let p1_remaining = self.insert_after(p1_scaled, Op::Binary { x: p1_orig, y: p1_scaled, bop: BOp::Sub });
        let p1_maxed = self.insert_after(p1_remaining, Op::Binary { x: p1_remaining, y: p1_nelt, bop: BOp::Max });
        let p1_sum = self.insert_after(p1_maxed, Op::Binary { x: p1_remaining, y: p1_nelt, bop: BOp::Add });
        let p1_loop_len = self.insert_after(p1_sum, Op::Binary { x: p1_sum, y: p1_maxed, bop: BOp::Sub });
        let loop_p1 = self.insert_after(p1_loop_len, Op::Loop { len: p1_loop_len });
        let p1_elem_idx = self.insert_after(loop_p1, Op::Binary { x: p1_scaled, y: loop_p1, bop: BOp::Add });
        let mut loop_end = p1_elem_idx;
        for &(lid, _src) in &load_ids {
            let local = in_locals[&lid];
            let dup = self.insert_after(
                loop_end,
                Op::Load {
                    src: match self.at(lid) {
                        Op::Load { src, .. } => *src,
                        _ => unreachable!(),
                    },
                    index: p1_elem_idx,
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

        // ── Phase 2: Tile compute ──
        // Insert tile compute + tile stores for all outputs before first_sid
        for &(sid, _dst) in &store_ids {
            let x = match self.at(sid) {
                Op::Store { x, .. } => *x,
                _ => unreachable!(),
            };
            let out_local = out_locals[&sid];
            let mut store_tile_map = tile_map.clone();
            let tile_result = self.clone_compute_tile(x, &mut store_tile_map, first_sid);
            self.insert_before(first_sid, Op::Store { dst: out_local, x: tile_result, index: c0, layout: TILE_T });
        }

        // Single barrier separating phase 2 and phase 3
        let barrier = self.insert_before(first_sid, Op::Barrier);

        // ── Phase 3: Single scalar local → global loop for all outputs ──
        // Compute loop bound = min(orig_len - gidx * TILE_NELT, TILE_NELT)
        // Uses: N - max(gidx*1024 + N - orig_len, 0)  [= N - max(overhang, 0)]
        // gidx*1024 = gidx*1025 - gidx  (avoids constant-fold to shift → CSE match with Phase 1)
        let p3_nelt = self.insert_after(barrier, Op::Const(Constant::idx(TILE_NELT as u64)));
        let p3_orig = self.insert_after(p3_nelt, Op::Const(Constant::idx(orig_len as u64)));
        let p3_1025 = self.insert_after(p3_orig, Op::Const(Constant::idx(1025u64)));
        let p3_zero = self.insert_after(p3_1025, Op::Const(Constant::idx(0u64)));
        let p3_scaled_extra = self.insert_after(p3_zero, Op::Binary { x: gidx, y: p3_1025, bop: BOp::Mul });
        let p3_scaled = self.insert_after(p3_scaled_extra, Op::Binary { x: p3_scaled_extra, y: gidx, bop: BOp::Sub });
        let p3_over = self.insert_after(p3_scaled, Op::Binary { x: p3_scaled, y: p3_nelt, bop: BOp::Add });
        let p3_over = self.insert_after(p3_over, Op::Binary { x: p3_over, y: p3_orig, bop: BOp::Sub });
        let p3_over_maxed = self.insert_after(p3_over, Op::Binary { x: p3_over, y: p3_zero, bop: BOp::Max });
        let p3_loop_len = self.insert_after(p3_over_maxed, Op::Binary { x: p3_nelt, y: p3_over_maxed, bop: BOp::Sub });
        let loop_p3 = self.insert_after(p3_loop_len, Op::Loop { len: p3_loop_len });
        let p3_elem_idx = self.insert_after(loop_p3, Op::Binary { x: p3_scaled, y: loop_p3, bop: BOp::Add });
        let mut body_last = p3_elem_idx;
        for &(sid, _dst) in &store_ids {
            let out_local = out_locals[&sid];
            body_last = self.insert_after(body_last, Op::Load { src: out_local, index: loop_p3, layout: MemLayout::Scalar });
            body_last = self
                .insert_after(body_last, Op::Store { dst: _dst, x: body_last, index: p3_elem_idx, layout: MemLayout::Scalar });
            self.remove_op(sid);
        }
        self.insert_after(body_last, Op::EndLoop);

        self.verify();
    }

    // ── helpers ──

    fn find_gidx(&self) -> Option<(OpId, Dim)> {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::GroupIndex { len, axis: 0 } = self.at(op_id) {
                if *len > 0 {
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
