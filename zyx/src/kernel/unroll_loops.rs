// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
#[allow(unused)]
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
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
        //println!("UNROLL len={} limit={}", len, len > 64);
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
        let Op::Loop { len } = self.ops[loop_id].op else { return };
        if factor < 2 || !len.is_multiple_of(factor) {
            return;
        }

        if self.ops.len().0 as u64 * factor > 5000 {
            return;
        }

        println!("Unroll tree reduce for loop={loop_id}, factor={factor}");

        // Find the acc
        let acc_id;
        let mut op_id = self.prev_op(loop_id);
        loop {
            if op_id.is_null() {
                return; // not a reduce, just a loop
            }
            match self.ops[op_id].op {
                Op::Loop { .. } => return, // nested reduce or no reduce
                Op::Define { scope: Scope::Register, .. } => {
                    acc_id = op_id;
                    break;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        };

        // Find the store to acc
        let mut op_id = acc_id;
        let acc_init;
        loop {
            if let Op::Store { dst, x, .. } = self.ops[op_id].op && dst == acc_id {
                acc_init = x;
                break;
            }
            op_id = self.next_op(op_id);
            if op_id == loop_id {
                unreachable!();
            }
        }

        let mut has_store = false;
        let mut op_id = self.next_op(loop_id);
        let endloop_id;
        loop {
            match self.ops[op_id].op {
                Op::Loop { .. } => return, // no nested loops
                Op::Store { vlen, .. } => {
                    if has_store || vlen != 1 {
                        return;
                    }
                    has_store = true;
                }
                Op::EndLoop => {
                    endloop_id = op_id;
                    break;
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        let mut map = Map::default();

        let new_loop = self.insert_before(loop_id, Op::Loop { len: len / factor });
        let mut op_id = self.next_op(loop_id);
        let stride = self.insert_before(loop_id, Op::Const(Constant::idx(factor)));
        self.ops[loop_id].op = Op::Binary { x: new_loop, y: stride, bop: BOp::Mul };
        let mut new_ones = Vec::with_capacity(factor as usize - 1);
        for i in 1..factor {
            let offset = self.insert_before(op_id, Op::Const(Constant::idx(i)));
            let new_id = self.insert_before(op_id, Op::Binary { x: loop_id, y: offset, bop: BOp::Add });
            new_ones.push(new_id);
        }
        map.insert(loop_id, new_ones);

        while op_id != endloop_id {
            let this_id = op_id;
            op_id = self.next_op(op_id);
            if let Op::Load { src, index: _, vlen: 1 } = self.ops[this_id].op && src == acc_id {
                // TODO debug assert index is const zero
                map.insert(this_id, vec![acc_init; factor as usize- 1]);
            } else if let Op::Store { dst, x, index, vlen: 1 } = self.ops[this_id].op && dst == acc_id {
                // TODO debug assert index is const zero
                let y = if let Some(mapping) = map.get(&x) {
                    mapping[0]
                } else {
                    x
                };
                let mut carry = this_id;
                self.ops[this_id].op = Op::Binary { x, y, bop: BOp::Add };
                for i in 1..factor - 1 {
                    let x = if let Some(mapping) = map.get(&x) {
                        mapping[i as usize]
                    } else {
                        x
                    };
                    carry = self.insert_before(op_id, Op::Binary { x, y: carry, bop: BOp::Add });
                }
                self.insert_before(op_id, Op::Store { dst, x: carry, index, vlen: 1 });
            } else {
                let mut new_ones = Vec::with_capacity(factor as usize - 1);
                for i in 1..factor {
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
        }

        self.verify();
    }
}
