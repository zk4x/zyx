// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
#[allow(unused)]
use crate::{
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    #[allow(unused)]
    pub fn opt_unroll(_: &Kernel) -> (Optimization, usize) {
        (Optimization::UnrollLoops { factors: vec![8, 4, 16, 2] }, 4)
    }

    #[allow(unused)]
    pub const fn opt_unroll_constant_loops(_: &Kernel) -> (Optimization, usize) {
        (Optimization::UnrollConstantLoops, 1)
    }

    pub fn eliminate_zero_len_index(&mut self) {
        for node in self.ops.values_mut() {
            if let Op::Index { len, .. } = node.op {
                if len == 1 {
                    node.op = Op::Const(Constant::idx(0));
                }
            }
        }
        self.verify();
    }

    pub fn unroll_loops(&mut self, unroll_dim: Dim) {
        let mut endloop_ids = Vec::new();
        let mut op_id = self.tail;
        while !op_id.is_null() {
            if self.ops[op_id].op == Op::EndLoop {
                endloop_ids.push(op_id);
            }
            if let Op::Loop { len, .. } = self.ops[op_id].op {
                let _ = endloop_ids.pop().unwrap();
                if len as usize <= unroll_dim as usize
                    && self.ops.len().0 as usize + (self.n_ops_in_loop(op_id) * (len as usize - 1)) < 5_000
                {
                    self.unroll_loop(op_id);
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
                    endloop_ids.pop().unwrap();
                    let is_const = constant_loops.pop().unwrap();
                    if !is_const {
                        if let Some(inner_loop) = constant_loops.last_mut() {
                            *inner_loop = false;
                        }
                    }
                    if len == 1
                        || (is_const && self.ops.len().0 as usize + (self.n_ops_in_loop(op_id) * (len as usize - 1)) < 5_000)
                    {
                        self.unroll_loop(op_id);
                    }
                }
                Op::Store { dst, .. } => {
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                Op::Load { src, .. } => {
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                _ => {}
            }
            op_id = prev;
        }
    }

    pub fn unroll_loop(&mut self, loop_id: OpId) {
        let Op::Loop { len } = self.ops[loop_id].op else { return };
        let len = len as usize;
        eprintln!("UNROLL len={} limit={}", len, len > 64);
        if len == 0 || len > 256 {
            return;
        }
        eprintln!("UNROLL doing it");

        let mut endloop_id = self.next_op(loop_id);
        while !matches!(self.ops[endloop_id].op, Op::EndLoop) {
            endloop_id = self.next_op(endloop_id);
        }

        let body_start = self.next_op(loop_id);
        let body_end = endloop_id;

        for iter in 1..len {
            let iter_op = self.insert_before(endloop_id, Op::Const(Constant::idx(iter as u64)));
            let mut op_id = body_start;
            while op_id != body_end {
                let mut new_op = self.ops[op_id].op.clone();
                for param in new_op.parameters_mut() {
                    if *param == loop_id {
                        *param = iter_op;
                    }
                }
                self.insert_before(endloop_id, new_op);
                op_id = self.next_op(op_id);
            }
        }
        self.ops[loop_id].op = Op::Const(Constant::idx(0));
        self.verify();
    }
}
