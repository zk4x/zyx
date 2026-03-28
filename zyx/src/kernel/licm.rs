// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::kernel::{Kernel, Op, OpId};
use crate::{Map, Set};

impl Kernel {
    pub fn swap_commutative(&mut self) {
        // Tracks whether a value depends on a loop index
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.at(op_id) {
                Op::Move { .. }
                | Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Reduce { .. } => unreachable!(),
                Op::Devectorize { .. } | Op::WMMA { .. } | Op::Vectorize { .. } => loop_depth,
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } => loop_dep[x],
                &Op::Binary { x, y, bop } => {
                    if bop.is_commutative() && !self.ops[x].op.is_const() {
                        if loop_dep[&x] > loop_dep[&y] || self.ops[y].op.is_const() || self.ops[x].op.is_load() {
                            //println!("Swapping {x}, {y}, loop dep {} > {}: {:?}, {:?}", loop_dep[&x], loop_dep[&y], self.ops[x].op, self.ops[y].op);
                            if let Op::Binary { x, y, .. } = &mut self.ops[op_id].op {
                                std::mem::swap(x, y);
                            }
                        }
                    }
                    loop_dep[&x].max(loop_dep[&y])
                }
                Op::Mad { x, y, z } => loop_dep[&x].max(loop_dep[&y]).max(loop_dep[&z]),
                Op::Index { .. } | Op::Load { .. } | Op::Store { .. } | Op::Const(_) | Op::Define { .. } => loop_depth,
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn reassociate_commutative(&mut self, _: u16) {
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.at(op_id) {
                Op::Move { .. }
                | Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Reduce { .. } => unreachable!(),
                Op::Vectorize { ops } => {
                    let mut max = 0;
                    for op in ops {
                        max = max.max(loop_dep[op]);
                    }
                    max
                }
                Op::Devectorize { .. } => todo!(),
                Op::Mad { x, y, z } => loop_dep[x].max(loop_dep[y]).max(loop_dep[z]),
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } => loop_dep[x],
                Op::Binary { x, y, .. } => loop_dep[x].max(loop_dep[y]),
                Op::Index { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::WMMA { .. } => loop_depth,
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        let mut op_id = self.head;
        'a: while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { bop, .. } = self.at(op_id) {
                if !bop.is_commutative() || !bop.is_associative() {
                    op_id = next;
                    continue 'a;
                }

                // Get all the leafs
                let mut params = vec![op_id];
                let mut chain = Vec::new();
                while let Some(param) = params.pop() {
                    if let &Op::Binary { x, y, bop: t_bop } = self.at(param) {
                        if t_bop == bop {
                            params.push(x);
                            params.push(y);
                            continue;
                        }
                    }
                    chain.push(param);
                    // We have to be somewhat reasonabe about those chains
                    if chain.len() > 20 {
                        op_id = next;
                        continue 'a;
                    }
                }
                if chain.len() < 2 {
                    op_id = next;
                    continue 'a;
                }
                chain.sort_by_key(|id| loop_dep[id]);

                // Rebuild chain
                let mut prev_acc = chain[0];
                let mut j = 1;
                while j < chain.len() - 1 {
                    let op = Op::Binary { x: chain[j], y: prev_acc, bop };
                    let new_acc = self.insert_before(op_id, op);
                    prev_acc = new_acc;
                    j += 1;
                }
                self.ops[op_id].op = Op::Binary { x: chain[j], y: prev_acc, bop };
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn loop_invariant_code_motion(&mut self) {
        let mut endloop_is = Vec::new();
        let mut loop_id = self.tail;
        while !loop_id.is_null() {
            if *self.at(loop_id) == Op::EndLoop {
                endloop_is.push(loop_id);
            }
            if let Op::Loop { .. } = self.at(loop_id) {
                let mut op_ids_in_loop = Set::default();
                op_ids_in_loop.insert(loop_id); // Loop op is the primary op that breaks LICM

                let mut op_id = loop_id;
                let endloop_id = endloop_is.pop().unwrap();
                while op_id != endloop_id {
                    let op = self.at(op_id);
                    let next_op_id = self.next_op(op_id);

                    if !matches!(
                        op,
                        Op::Store { .. } | Op::Load { .. } | Op::Loop { .. } | Op::EndLoop | Op::Define { .. }
                    ) && op.parameters().all(|op_id| !op_ids_in_loop.contains(&op_id))
                    {
                        self.move_op_before(op_id, loop_id);
                    } else {
                        op_ids_in_loop.insert(op_id);
                    }

                    op_id = next_op_id;
                }
            }
            loop_id = self.prev_op(loop_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }
}
