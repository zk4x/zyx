// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused)]
use crate::{
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
    Map,
};

impl Kernel {
    pub fn opt_unroll_config(&self) -> u16 {
        4
    }

    pub fn opt_unroll(&mut self, config: u16) {
        let unroll_dim = [2, 4, 8, 16][config as usize];
        self.unroll_loops(unroll_dim);
    }

    pub fn unroll_loops(&mut self, unroll_dim: usize) {
        let mut endloop_ids = Vec::new();
        let mut op_id = self.tail;
        while !op_id.is_null() {
            if self.ops[op_id].op == Op::EndLoop {
                endloop_ids.push(op_id);
            }
            if let Op::Loop { len, .. } = self.ops[op_id].op {
                let endloop_id = endloop_ids.pop().unwrap();
                if len <= unroll_dim && self.ops.len().0 as usize + (self.n_ops_in_loop(op_id) * (len - 1)) < 5_000 {
                    self.unroll_loop(op_id, endloop_id, len);
                }
            }
            op_id = self.prev_op(op_id);
        }
    }

    fn n_ops_in_loop(&self, loop_id: OpId) -> usize {
        let mut op_id = self.next_op(loop_id);
        let mut n_loops = 1;
        let mut n_ops = 0;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    n_loops += 1;
                }
                Op::EndLoop => {
                    n_loops -= 1;
                    if n_loops == 0 {
                        return n_ops;
                    }
                }
                _ => {}
            }
            n_ops += 1;
            op_id = self.next_op(op_id);
        }
        n_ops
    }

    pub fn unroll_constant_loops(&mut self) {
        let mut endloop_ids = Vec::new();
        let mut op_id = self.tail;
        let mut constant_loops = vec![true];
        while !op_id.is_null() {
            let prev = self.prev_op(op_id);
            match self.ops[op_id].op {
                Op::EndLoop => {
                    endloop_ids.push(op_id);
                    constant_loops.push(true);
                }
                Op::Loop { len, .. } => {
                    let endloop_id = endloop_ids.pop().unwrap();
                    let is_const = constant_loops.pop().unwrap();
                    if !is_const {
                        if let Some(inner_loop) = constant_loops.last_mut() {
                            *inner_loop = false;
                        }
                    }
                    if len == 1
                        || (is_const && self.ops.len().0 as usize + (self.n_ops_in_loop(op_id) * (len - 1)) < 5_000)
                    {
                        self.unroll_loop(op_id, endloop_id, len);
                    }
                }
                Op::Store { dst, .. } => {
                    let Op::Define { scope, .. } = self.ops[dst].op else {
                        unreachable!()
                    };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                Op::Load { src, .. } => {
                    let Op::Define { scope, .. } = self.ops[src].op else {
                        unreachable!()
                    };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                _ => {}
            }
            op_id = prev;
        }
    }

    pub fn unroll_loop(&mut self, loop_id: OpId, endloop_id: OpId, dim: usize) {
        self.ops[loop_id].op = Op::Const(Constant::idx(0));
        let last_loop_op = self.prev_op(endloop_id);

        for idx in 1..dim {
            let mut new_ops_map = Map::default();
            let idx_op = self.insert_before(endloop_id, Op::Const(Constant::idx(idx as u64)));
            new_ops_map.insert(loop_id, idx_op);

            let mut op_id = self.next_op(loop_id);
            loop {
                let mut op = self.ops[op_id].op.clone();
                for param in op.parameters_mut() {
                    if let Some(&new_param) = new_ops_map.get(param) {
                        *param = new_param;
                    }
                }
                let new_op_id = self.insert_before(endloop_id, op);
                new_ops_map.insert(op_id, new_op_id);

                if op_id == last_loop_op {
                    break;
                }
                op_id = self.next_op(op_id);
            }
        }
        self.remove_op(endloop_id);

        #[cfg(debug_assertions)]
        self.verify();
    }
}
