// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    /// Pads a global index by the given length, guarding out-of-range loads.
    ///
    /// This extends `Op::Index { len: current_len, .. }` to `len: current_len + pad_len`
    /// and inserts guards around every `Op::Load` whose index depends on `gidx_id`.
    /// For threads whose coordinate ≥ `current_len`, the loaded value is replaced
    /// with `pad_value` and the memory access is redirected to element 0 (safe).
    ///
    /// This is useful for tiling: when a tensor size isn't a multiple of the tile
    /// size, pad the index so the grid covers full tiles, and padded positions
    /// read the pad value (typically zero, or the neutral element for reductions).
    ///
    /// # Panics
    /// - If `gidx_id` is not an `Op::Index` node.
    /// - If a load's source is not an `Op::Define` (internal consistency).
    pub fn pad_index(&mut self, gidx_id: OpId, current_len: Dim, pad_len: Dim, pad_value: Constant) {
        if pad_len == 0 {
            return;
        }

        // 1. Extend the index length
        let Op::Index { len, .. } = &mut self.ops[gidx_id].op else {
            panic!("pad_index: op is not an Index");
        };
        *len = current_len + pad_len;

        // 2. Create shared constants before the index
        let limit = self.insert_before(gidx_id, Op::Const(Constant::idx(current_len)));
        let pad = self.insert_before(gidx_id, Op::Const(pad_value));

        // 3. Walk all ops to find loads depending on this index
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Load { src, index: load_idx, vlen } = self.ops[op_id].op.clone() {
                debug_assert_eq!(vlen, 1, "pad_index must run before any upcast pass");
                if self.depends_on(load_idx, gidx_id, &mut Set::default()) {
                    let dtype = match self.at(src) {
                        Op::Define { dtype, .. } => *dtype,
                        _ => unreachable!(),
                    };

                    // cond = gidx < current_len
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    // Cast condition to index type for offset zeroing
                    let cast_idx = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: load_idx, y: cast_idx, bop: BOp::Mul });
                    let safe_load = self.insert_before(op_id, Op::Load { src, index: safe_idx, vlen });
                    // Cast condition to data type for result selection
                    let cond_dt = self.insert_before(op_id, Op::Cast { x: cond, dtype });
                    // (loaded - pad) * cond + pad
                    let sub = self.insert_before(op_id, Op::Binary { x: safe_load, y: pad, bop: BOp::Sub });
                    let result = self.insert_before(op_id, Op::Mad { x: sub, y: cond_dt, z: pad });
                    // Remap consumers from the old load to the result
                    self.remap(op_id, result);
                }
            }
            op_id = next;
        }
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
