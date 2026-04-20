// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::Map;
use crate::kernel::{Kernel, Op, OpId, Scope};

impl Kernel {
    #[allow(unused)]
    pub fn opt_register_tiling(&self) -> (Optimization, usize) {
        let candidates: Vec<u64> = vec![2, 4, 8, 16];
        let mut global_upcasts = Map::default();
        let mut reduce_factors = Map::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 16 {
                    let applicable: Vec<u64> = candidates
                        .iter()
                        .copied()
                        .filter(|&f| len.is_multiple_of(f) && len / f >= 4)
                        .collect();
                    if !applicable.is_empty() {
                        reduce_factors.insert(op_id, applicable);
                    }
                }
            }
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global && len >= 8 {
                    let applicable: Vec<u64> = candidates
                        .iter()
                        .copied()
                        .filter(|&f| len.is_multiple_of(f) && len / f >= 4)
                        .collect();
                    if !applicable.is_empty() {
                        global_upcasts.insert(op_id, applicable);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        if global_upcasts.is_empty() || reduce_factors.is_empty() {
            return (
                Optimization::RegisterTiling { reduce_splits: reduce_factors, global_upcasts },
                0,
            );
        }

        let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();
        let n_reduce_options: usize = reduce_factors.values().map(Vec::len).product();

        let n_configs = n_global_options * n_reduce_options;
        (
            Optimization::RegisterTiling { reduce_splits: reduce_factors, global_upcasts },
            n_configs,
        )
    }

    #[allow(unused)]
    pub fn apply_register_tiling(
        &mut self,
        reduce_splits: &Map<OpId, Vec<u64>>,
        global_upcasts: &Map<OpId, Vec<u64>>,
        config: usize,
    ) {
        let n_global = global_upcasts.len();
        let n_reduce = reduce_splits.len();
        if n_global == 0 || n_reduce == 0 {
            return;
        }

        let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();

        let mut remaining_global = config % n_global_options;
        let mut remaining_reduce = config / n_global_options;

        for (reduce_id, factors) in reduce_splits.iter() {
            let n_options = factors.len();
            let factor_idx = remaining_reduce % n_options;
            remaining_reduce /= n_options;
            let reduce_factor = factors[factor_idx];

            let Op::Loop { len, .. } = self.ops[*reduce_id].op else {
                continue;
            };
            let original_len = len;

            self.split_dim(
                *reduce_id,
                vec![
                    Op::Loop { len: original_len / reduce_factor },
                    Op::Loop { len: reduce_factor },
                ],
            );
        }

        let mut new_global_upcasts = Vec::new();
        for (_, factors) in global_upcasts.iter() {
            let n_options = factors.len() + 1;
            let factor_idx = remaining_global % n_options;
            remaining_global /= n_options;

            let factor = if factor_idx == 0 { 1 } else { factors[factor_idx - 1] };
            new_global_upcasts.push(factor);
        }

        let mut idx = 0;
        for (op_id, _) in global_upcasts.iter() {
            let factor = new_global_upcasts[idx];
            if factor > 1 {
                self.upcast(*op_id, factor as u64);
            }
            idx += 1;
        }
    }
}
