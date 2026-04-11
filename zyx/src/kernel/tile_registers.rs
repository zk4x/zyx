// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::kernel::{Kernel, Op, Scope};
use crate::{Map, Set};

impl Kernel {
    #[allow(unused)]
    pub fn opt_register_tiling(&self) -> (Optimization, usize) {
        let candidates: Vec<u64> = vec![2, 4, 8, 16];
        let mut global_upcasts = Map::default();
        let mut reduce_factors = Map::default();
        let mut reduce_ids = Set::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 16 {
                    let applicable: Vec<u64> = candidates.iter().copied().filter(|&f| len.is_multiple_of(f) && len / f >= 4).collect();
                    if !applicable.is_empty() {
                        reduce_factors.insert(op_id, applicable);
                        reduce_ids.insert(op_id);
                    }
                }
            }
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global && len >= 8 {
                    let applicable: Vec<u64> = candidates.iter().copied().filter(|&f| len.is_multiple_of(f) && len / f >= 4).collect();
                    if !applicable.is_empty() {
                        global_upcasts.insert(op_id, applicable);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        if global_upcasts.is_empty() || reduce_factors.is_empty() {
            return (Optimization::RegisterTiling { reduce_splits: reduce_factors, global_upcasts }, 0);
        }

        let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();
        let n_reduce_options: usize = reduce_factors.values().map(|v| v.len()).product();

        let n_configs = n_global_options * n_reduce_options;
        (Optimization::RegisterTiling { reduce_splits: reduce_factors, global_upcasts }, n_configs)
    }
}
