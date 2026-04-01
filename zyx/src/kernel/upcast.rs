// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::kernel::{Kernel, Op, OpId, Scope};

impl Kernel {
    pub fn opt_upcast(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                let mut r_factors: Vec<usize> = vec![4, 8, 2, 16];
                if scope == Scope::Global {
                    r_factors.retain(|&f| len.is_multiple_of(f) && len / f >= 4);
                    for &f in &r_factors {
                        factors.push((op_id, f));
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len() as usize;
        (Optimization::Upcast { factors }, n_configs)
    }

    pub fn upcast(&mut self, op_id: OpId, factor: usize) {
        let Op::Index { len, scope, axis } = self.ops[op_id].op else {
            unreachable!("upcast only works on Index ops");
        };
        debug_assert_eq!(scope, Scope::Global);
        debug_assert!(len.is_multiple_of(factor));

        self.split_dim(
            op_id,
            vec![
                Op::Index {
                    len: len / factor,
                    scope: Scope::Global,
                    axis,
                },
                Op::Loop { len: factor },
            ],
        );

        let mut upcast_loop_id = OpId::NULL;
        let mut reduce_loop_id = OpId::NULL;

        let mut op_id_iter = self.head;
        while !op_id_iter.is_null() {
            if let Op::Loop { len, .. } = self.ops[op_id_iter].op {
                if len == factor && upcast_loop_id == OpId::NULL {
                    upcast_loop_id = op_id_iter;
                } else if len != factor {
                    reduce_loop_id = op_id_iter;
                }
            }
            op_id_iter = self.next_op(op_id_iter);
        }

        if reduce_loop_id != OpId::NULL {
            self.jam_loop(upcast_loop_id, reduce_loop_id);
        }

        // self.verify();
    }
}
