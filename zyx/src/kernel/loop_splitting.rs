// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::kernel::{Kernel, Op, Scope};

impl Kernel {
    pub fn opt_split_global_to_local(&self) -> (Optimization, usize) {
        let mut op_id = self.head;
        let mut factors = Vec::new();
        let mut seen_axes = crate::Map::default();
        while !op_id.is_null() {
            if let Op::Index { len, scope, axis } = self.ops[op_id].op {
                let mut l_factors: Vec<usize> = vec![32, 64, 16, 8, 4, 2];
                if scope == Scope::Global {
                    l_factors.retain(|&f| len.is_multiple_of(f));
                    for &f in &l_factors {
                        factors.push((op_id, f));
                    }
                    seen_axes.insert(axis, op_id);
                }
                if scope == Scope::Local {
                    if let Some(global_id) = seen_axes.get(&axis) {
                        factors.retain(|(op_id, _)| global_id != op_id);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::SplitGlobalToLocal { factors }, n_configs)
    }

    pub fn opt_split_loop(&self) -> (Optimization, usize) {
        let candidates = vec![8, 16, 4, 2];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 16 {
                    for &factor in &candidates {
                        if len.is_multiple_of(factor) {
                            factors.push((op_id, factor));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::SplitLoop { factors }, n_configs)
    }
}
