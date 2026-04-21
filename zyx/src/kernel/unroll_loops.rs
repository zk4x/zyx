// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
#[allow(unused)]
use crate::{
    Map,
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
        println!("UNROLL len={} limit={}", len, len > 64);
        if len == 0 || len > 64 {
            return;
        }

        let mut loop_depth = 1;
        let mut endloop_id = self.next_op(loop_id);
        while !endloop_id.is_null() {
            match self.ops[endloop_id].op {
                Op::Loop { .. } => {
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            endloop_id = self.next_op(endloop_id);
        }

        /*
        // This is a skeleton of interleaved unroll
        // TODO fix the load/store interleaving issue later if it's worth it
        let mut endloop_id = self.next_op(loop_id);
        while !matches!(self.ops[endloop_id].op, Op::EndLoop) {
            // For now just don't unroll if there are inner loops
            if matches!(self.ops[endloop_id].op, Op::Loop { .. }) {
                return;
            }
            endloop_id = self.next_op(endloop_id);
        }

        let mut map = Map::default();

        let mut op_id = self.next_op(loop_id);
        self.ops[loop_id].op = Op::Const(Constant::idx(0));
        let mut new_ones = Vec::with_capacity(len as usize - 1);
        for i in 1..len {
            let new_id = self.insert_before(op_id, Op::Const(Constant::idx(i)));
            new_ones.push(new_id);
        }
        map.insert(loop_id, new_ones);

        while op_id != endloop_id {
            let this_id = op_id;
            op_id = self.next_op(op_id);
            let mut new_ones = Vec::with_capacity(len as usize - 1);
            for i in 1..len {
                let mut new_op = self.ops[this_id].op.clone();
                for param in new_op.parameters_mut() {
                    if let Some(mapping) = map.get(param) {
                        *param = mapping[i as usize - 1];
                    }
                }
                let new_id = self.insert_before(op_id, new_op);
                new_ones.push(new_id);
            }
            map.insert(this_id, new_ones);
        }
        self.remove_op(endloop_id);*/

        self.ops[loop_id].op = Op::Const(Constant::idx(0));
        let last_loop_op = self.prev_op(endloop_id);

        for idx in 1..len {
            let mut new_ops_map = Map::default();
            let idx_op = self.insert_before(endloop_id, Op::Const(Constant::idx(idx)));
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


        self.verify();
    }

    /// Unrolls loop:
    /// acc a
    /// for 0..32 {
    ///     z = ...
    ///     x = load a
    ///     x1 = x + z
    ///     store a <- x1
    /// }
    ///
    /// Using tree reduce unroll to:
    ///
    /// acc a
    /// for 0..8 {
    ///     z = ...
    ///     z1 = ...
    ///     z2 = ...
    ///     z3 = ...
    ///     x = load a
    ///     x1 = x + z
    ///     x2 = x1 + z1
    ///     x3 = x2 + z2
    ///     x4 = x3 + z3
    ///     store a <- x4
    /// }
    ///
    pub fn unroll_tree_reduce(&mut self, loop_id: OpId, factor: Dim) {
        self.debug_colorless();
        println!("Unroll tree reduce on the above kernel, loop_id={loop_id}, factor={factor}");

        self.verify();
    }
}
