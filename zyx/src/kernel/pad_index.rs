// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope},
    shape::Dim,
};

use super::autotune::Optimization;

impl Kernel {
    /// Pads a global index to the next multiple of `tile_size`, guarding out-of-range loads
    /// and skipping out-of-range stores.
    ///
    /// This extends `Op::Index { len: current_len, .. }` to `len: current_len + pad_len`
    /// so the grid covers full tiles.  OOB reads are redirected to element 0 (safe).
    /// OOB stores are wrapped in `Op::If { .. }` / `Op::EndIf` and skipped entirely.
    ///
    /// Useful for tiling: when a tensor dimension isn't a multiple of the tile size,
    /// pad the index so the grid covers full tiles, and OOB threads compute garbage
    /// but never write it to memory.
    ///
    /// # Panics
    /// - If `gidx_id` is not an `Op::Index` node.
    pub fn pad_index(&mut self, gidx_id: OpId, current_len: Dim, pad_len: Dim, _pad_value: Constant) {
        if pad_len == 0 {
            return;
        }

        // 1. Extend the index length
        let Op::Index { len, .. } = &mut self.ops[gidx_id].op else {
            panic!("pad_index: op is not an Index");
        };
        *len = current_len + pad_len;

        // 2. Create limit constant for comparison
        let limit = self.insert_before(gidx_id, Op::Const(Constant::idx(current_len)));

        // 3. Walk all ops to guard loads and stores depending on this index
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            // Guard stores: skip OOB stores entirely
            if let Op::Store { index: store_idx, .. } = self.ops[op_id].op.clone() {
                if self.depends_on(store_idx, gidx_id, &mut Set::default()) {
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    self.insert_before(op_id, Op::If { condition: cond });
                    self.insert_after(op_id, Op::EndIf);
                }
            }

            // Guard loads: redirect OOB reads to element 0 (safe)
            if let Op::Load { src, index: load_idx, vlen } = self.ops[op_id].op.clone() {
                debug_assert_eq!(vlen, 1, "pad_index must run before any upcast pass");
                if self.depends_on(load_idx, gidx_id, &mut Set::default()) {
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    let cast_idx = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: load_idx, y: cast_idx, bop: BOp::Mul });
                    let safe_load = self.insert_before(op_id, Op::Load { src, index: safe_idx, vlen });
                    self.remap(op_id, safe_load);
                    self.remove_op(op_id);
                }
            }

            op_id = next;
        }
    }

    pub fn opt_pad_index(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope: Scope::Global, .. } = self.ops[op_id].op {
                if len % 32 != 0 {
                    factors.push((op_id, 32 - (len % 32)));
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::PadIndex { factors }, n_configs)
    }

    fn depends_on(&self, expr: OpId, target: OpId, visited: &mut Set<OpId>) -> bool {
        if expr == target || !visited.insert(expr) {
            return expr == target;
        }
        match self.at(expr) {
            Op::Const(_) | Op::Index { .. } | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop => false,
            op => op.parameters().any(|p| self.depends_on(p, target, visited)),
        }
    }
}
