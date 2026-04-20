// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::kernel::{Kernel, Op, OpId, Scope};

impl Kernel {
    #[allow(unused)]
    pub fn opt_register_tiling(&self) -> (Optimization, usize) {
        let candidates: Vec<u64> = vec![1, 2, 4, 8, 16, 32];
        let mut global_ids: Vec<(OpId, u64)> = Vec::new();
        let mut loop_id: Option<OpId> = None;
        let mut loop_len: u64 = 0;

        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global && len >= 8 {
                    let applicable: Vec<u64> = candidates.iter().copied().filter(|&f| len.is_multiple_of(f)).collect();
                    if !applicable.is_empty() {
                        global_ids.push((op_id, applicable.len() as u64));
                    }
                }
            }
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 16 {
                    loop_id = Some(op_id);
                    loop_len = len;
                }
            }
            op_id = self.next_op(op_id);
        }

        if global_ids.is_empty() || loop_id.is_none() {
            return (
                Optimization::RegisterTiling { global_ids: vec![], loop_id: OpId::NULL, loop_factors: vec![] },
                0,
            );
        }

        let applicable: Vec<u64> = candidates
            .iter()
            .copied()
            .filter(|&f| loop_len.is_multiple_of(f) && loop_len / f >= 4)
            .collect();

        if applicable.is_empty() {
            return (
                Optimization::RegisterTiling { global_ids: vec![], loop_id: OpId::NULL, loop_factors: vec![] },
                0,
            );
        }

        let applicable: Vec<u64> = candidates
            .iter()
            .copied()
            .filter(|&f| loop_len.is_multiple_of(f) && loop_len / f >= 4)
            .collect();

        if applicable.is_empty() {
            return (
                Optimization::RegisterTiling { global_ids: vec![], loop_id: OpId::NULL, loop_factors: vec![] },
                0,
            );
        }

        let n_global_options: usize = global_ids.iter().map(|&(_, n)| n as usize).product();
        let n_loop_options = applicable.len();
        let n_configs = n_global_options * n_loop_options;

        (
            Optimization::RegisterTiling { global_ids, loop_id: loop_id.unwrap(), loop_factors: applicable },
            n_configs,
        )
    }

    pub fn register_tiling(&mut self, global_ids: Vec<(OpId, u64)>, loop_id: OpId, loop_factors: Vec<u64>, config: usize) {
        let n_global_options: usize = global_ids.iter().map(|&(_, n)| n as usize).product();
        let remaining_global = config % n_global_options;
        let remaining_loop = config / n_global_options;

        let loop_factor = loop_factors[remaining_loop];
        if loop_factor > 1 {
            let Op::Loop { len, .. } = self.ops[loop_id].op else { return };
            let original_len = len;
            self.split_dim(
                loop_id,
                vec![Op::Loop { len: original_len / loop_factor }, Op::Loop { len: loop_factor }],
            );
        }
        self.run_always_on_optimizations();

        let mut remaining = remaining_global;
        let upcast_factors: Vec<u64> = vec![1, 2, 4, 8, 16, 32];
        for (idx_id, n_options) in global_ids.iter() {
            let factor_idx = remaining % *n_options as usize;
            remaining /= *n_options as usize;
            let upcast_factor = upcast_factors[factor_idx];
            if upcast_factor > 1 {
                self.upcast(*idx_id, upcast_factor);
            }
        }

        self.unroll_loops(loop_factor);

        self.run_always_on_optimizations();
    }
}
