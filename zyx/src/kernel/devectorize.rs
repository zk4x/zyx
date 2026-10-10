// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Devectorize: unfold leftover unroll ranges back to scalar statements.
//!
//! Inverse tail of split-based coarsening (`coarsen` divides a group range
//! into a short group plus a [`RangeKind::Unroll`] range of lanes with no
//! body duplication). Tile matchers consume matched lanes into tile ops;
//! whatever lanes remain unfold here, one scalar statement per lane, so
//! every transport backend only ever sees scalar IR.
//!
//! The unfolding walk is the duplication walk the old duplicating `coarsen`
//! ran, with the lane constants (`0..width`) in place of its index offsets
//! and without its group splitting, storage growth, or accumulator
//! special-casing (the split already grew storages and lane-parameterized
//! the accumulator indices, so the generic operand-substitution arm covers
//! every op, including [`Op::GEP`] index trees).

use crate::{
    Map,
    dtype::Constant,
    kernel::{Kernel, MemLayout, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

impl Kernel {
    /// Unfold every used `RangeKind::Unroll` range into one scalar statement
    /// per lane. Unused ranges are removed. Runs in the device epilogue,
    /// after tile matching, before reader/writer publishing.
    pub fn devectorize_unroll(&mut self) {
        for _ in 0..10000 {
            let found =
                self.iter_unordered().find_map(|(op, x)| matches!(x, Op::Range { kind: RangeKind::Unroll(_), .. }).then_some(op));
            let Some(unroll_id) = found else { break };
            // A refused range (barrier, vectorized layout, size cap) stays
            // in place — stop instead of looping on it forever.
            if !self.devectorize_unroll_one(unroll_id) {
                break;
            }
        }
        self.verify();
    }

    /// Unfold a single unroll range: duplicate every lane-varying op once
    /// per lane with the range remapped to that lane's constant, rewrite
    /// the originals in place as lane 0, then remove the range. Returns
    /// false when the guards refuse (kernel left untouched).
    fn devectorize_unroll_one(&mut self, unroll_id: OpId) -> bool {
        let Op::Range { kind: RangeKind::Unroll(width), .. } = self.ops[unroll_id].op else {
            unreachable!("devectorize: not an unroll range")
        };
        let factor = width as u64;

        // Same guards as the old duplicating coarsen: refuse vectorized
        // kernels (non-scalar GEP layouts) and barriers.
        if self.ops.values().any(|node| match node.op {
            Op::Load { src } => match self.ops[src].op {
                Op::GEP { layout, .. } => layout != MemLayout::Scalar,
                _ => false,
            },
            Op::Store { dst, .. } => match self.ops[dst].op {
                Op::GEP { layout, .. } => layout != MemLayout::Scalar,
                _ => false,
            },
            Op::Copy { src, dst } => {
                !matches!(self.ops[src].op, Op::GEP { layout: MemLayout::Scalar, .. })
                    || !matches!(self.ops[dst].op, Op::GEP { layout: MemLayout::Scalar, .. })
            }
            Op::Barrier => true,
            _ => false,
        }) {
            return false;
        }

        if self.ops.len().0 as Dim * factor as Dim > 10000 {
            return false;
        }

        // Lane constants, lane 0 first (lane 0 rewrites the originals).
        let lane0 = self.insert_before(unroll_id, Op::Const(Constant::idx(0)));
        let mut lanes = Vec::with_capacity((factor - 1) as usize);
        for i in 1..factor {
            lanes.push(self.insert_before(unroll_id, Op::Const(Constant::idx(i))));
        }

        // First skip ops that don't need duplication
        let mut op_id = self.head;
        while !op_id.is_null()
            && matches!(
                self.ops[op_id].op,
                Op::Storage { scope: MemScope::Global | MemScope::Local, .. }
                    | Op::Param { .. }
                    | Op::Range { .. }
                    | Op::Const(_)
            )
        {
            op_id = self.next_op(op_id);
        }

        // For remapping parameters
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();

        // Now loop over remaining ops and duplicate lane-varying ones
        while !op_id.is_null() {
            let next_op_id = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Storage { .. } | Op::Range { .. } | Op::Loop { .. } | Op::EndLoop | Op::Barrier => {}
                ref op => {
                    let mut op = op.clone();
                    let mut varies = false;
                    for param in op.parameters_mut() {
                        if *param == unroll_id || remaps.contains_key(&*param) {
                            varies = true;
                            break;
                        }
                    }
                    if varies {
                        let mut ids = Vec::with_capacity((factor - 1) as usize);
                        let mut id = op_id;
                        for i in 0..(factor - 1) as usize {
                            let mut lane_op = op.clone();
                            // Reindex the op
                            for param in lane_op.parameters_mut() {
                                if *param == unroll_id {
                                    *param = lanes[i];
                                } else if let Some(remap) = remaps.get(param) {
                                    *param = remap[i];
                                }
                            }
                            id = self.insert_after(id, lane_op);
                            ids.push(id);
                        }
                        // Lane 0 rewrites the original in place (the unroll
                        // range has no inherent lane-0 meaning to inherit).
                        for param in self.ops[op_id].op.parameters_mut() {
                            if *param == unroll_id {
                                *param = lane0;
                            }
                        }
                        // Store which remaps will be used in the future
                        remaps.insert(op_id, ids);
                    }
                }
            }
            op_id = next_op_id;
        }

        self.remove_op(unroll_id);
        true
    }
}
