// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
};

impl Kernel {
    /// Duplicate an op, inserting the copy right after the given position.
    fn dup_after(&mut self, after_id: OpId, orig_id: OpId) -> OpId {
        let op = self.ops[orig_id].op.clone();
        self.insert_after(after_id, op)
    }

    pub fn opt_upcast(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global {
                    for &f in &[4, 8, 2, 16] {
                        if len.is_multiple_of(f) && len / f >= 4 {
                            factors.push((op_id, f));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::Upcast { factors }, n_configs)
    }

    pub fn upcast(&mut self, op_id: OpId, factor: usize) {
        debug_assert!(matches!(self.ops[op_id].op, Op::Index { scope: Scope::Global, .. }));

        if !self.ops.values().any(|node| matches!(node.op, Op::Loop { .. })) {
            let Op::Index { len, scope, axis } = self.ops[op_id].op else {
                return;
            };
            self.split_dim(
                op_id,
                vec![
                    Op::Index { len: len / factor, scope, axis },
                    Op::Index { len: factor, scope: Scope::Local, axis },
                ],
            );
            return;
        }

        // === UPCAST WITH REDUCE LOOPS === //
        let Op::Index { len, scope, axis } = self.ops[op_id].op else {
            unreachable!()
        };
        debug_assert!(len.is_multiple_of(factor));
        self.ops[op_id].op = Op::Index { len: len / factor, scope, axis };

        // Collect all non-trivial ops in order
        let mut ops_to_dup: Vec<OpId> = Vec::new();
        let mut id = self.head;
        while !id.is_null() {
            match self.ops[id].op {
                Op::Index { .. } | Op::Const(_) | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop => {}
                _ => ops_to_dup.push(id),
            }
            id = self.next_op(id);
        }

        // Find accumulator defines (mutable defines used by Load/Store)
        // Only register/local accumulators need index remapping
        let mut acc_defines: Set<OpId> = Set::default();
        for &dup_id in &ops_to_dup {
            match self.ops[dup_id].op {
                Op::Load { src, .. } => {
                    if let Op::Define { ro: false, scope: Scope::Register | Scope::Local, .. } = self.ops[src].op {
                        acc_defines.insert(src);
                    }
                }
                Op::Store { dst, .. } => {
                    if let Op::Define { ro: false, scope: Scope::Register | Scope::Local, .. } = self.ops[dst].op {
                        acc_defines.insert(dst);
                    }
                }
                _ => {}
            }
        }

        // Increase accumulator sizes by factor
        for &acc_id in &acc_defines {
            if let Op::Define { len, .. } = &mut self.ops[acc_id].op {
                *len *= factor;
            }
        }

        // Pre-create accumulator index helpers at the beginning (just constants 0..factor-1)
        let first_non_trivial = ops_to_dup[0];
        let mut acc_index_helpers: Map<OpId, Vec<OpId>> = Map::default();
        for &acc_id in &acc_defines {
            let mut helpers = Vec::with_capacity(factor);
            for i in 0..factor {
                let ic = self.insert_before(first_non_trivial, Op::Const(Constant::idx(i as u64)));
                helpers.push(ic);
            }
            acc_index_helpers.insert(acc_id, helpers);
        }

        // Process ops in forward order
        let mut remap: Map<OpId, Vec<OpId>> = Map::default();

        for &orig_id in &ops_to_dup {
            let orig_op = self.ops[orig_id].op.clone();
            let mut copies = Vec::with_capacity(factor);

            for i in 0..factor {
                let mut new_op = orig_op.clone();

                // Remap gidx - insert helpers right before this copy (inside the loop)
                for param in new_op.parameters_mut() {
                    if *param == op_id {
                        let insert_point = if i == 0 { orig_id } else { *copies.last().unwrap() };
                        let fc = self.insert_before(insert_point, Op::Const(Constant::idx(factor as u64)));
                        let ic = self.insert_before(insert_point, Op::Const(Constant::idx(i as u64)));
                        let mul = self.insert_before(insert_point, Op::Binary { x: op_id, y: fc, bop: crate::kernel::BOp::Mul });
                        let add = self.insert_before(insert_point, Op::Binary { x: mul, y: ic, bop: crate::kernel::BOp::Add });
                        *param = add;
                    } else if let Some(mapped) = remap.get(param) {
                        *param = mapped[i];
                    }
                }

                // Fix accumulator indexing
                if let Op::Load { src, index, .. } = &new_op {
                    if acc_defines.contains(src) {
                        if let Some(helpers) = acc_index_helpers.get(src) {
                            if let Op::Load { index: li, .. } = &mut new_op {
                                *li = helpers[i];
                            }
                        }
                    }
                }
                if let Op::Store { dst, index, .. } = &new_op {
                    if acc_defines.contains(dst) {
                        if let Some(helpers) = acc_index_helpers.get(dst) {
                            if let Op::Store { index: si, .. } = &mut new_op {
                                *si = helpers[i];
                            }
                        }
                    }
                }

                let new_id = if i == 0 {
                    self.ops[orig_id].op = new_op;
                    orig_id
                } else {
                    self.insert_after(*copies.last().unwrap(), new_op)
                };
                copies.push(new_id);
            }

            remap.insert(orig_id, copies);
        }

        self.verify();
        self.debug_colorless();
    }
}
