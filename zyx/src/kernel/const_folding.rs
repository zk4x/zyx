// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope, UOp},
};

impl Kernel {
    // Constant folding and deletion of useless ops, etc.
    pub fn constant_folding(&mut self, _: u16) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Move { .. } | Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => todo!(),
                Op::WMMA { .. }
                | Op::Vectorize { .. } // TODO
                | Op::Devectorize { .. } // TODO
                | Op::Store { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::Load { .. }
                | Op::Index { .. }
                | Op::Loop { .. }
                | Op::EndLoop => {}
                Op::Cast { x, dtype } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.cast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.unary(uop));
                    }
                }
                Op::Binary { x, y, bop } => match (self.at(x).clone(), self.at(y).clone()) {
                    (Op::Const(cx), Op::Const(cy)) => {
                        self.ops[op_id].op = Op::Const(Constant::binary(cx, cy, bop));
                    }
                    (Op::Const(cx), _) => match bop {
                        BOp::And if cx.dtype() == DType::Bool => self.remap(op_id, y),
                        BOp::Add if cx.is_zero() => self.remap(op_id, y),
                        BOp::Sub if cx.is_zero() => self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Neg },
                        BOp::Mul if cx.is_zero() => self.ops[op_id].op = Op::Const(cx.dtype().zero_constant()),
                        BOp::Mul if cx.is_one() => self.remap(op_id, y),
                        BOp::Mul if cx.is_two() => self.ops[op_id].op = Op::Binary { x: y, y, bop: BOp::Add },
                        BOp::Mul if cx.is_power_of_two() && cx.dtype() == IDX_T => {
                            let c = self.insert_before(op_id, Op::Const(cx.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x: y, y: c, bop: BOp::BitShiftLeft };
                        }
                        BOp::Div if cx.is_zero() => self.ops[op_id].op = Op::Const(cx.dtype().zero_constant()),
                        BOp::Div if cx.is_one() => self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Reciprocal },
                        BOp::Pow if cx.is_one() => self.ops[op_id].op = Op::Const(cx.dtype().one_constant()),
                        BOp::Max if cx.is_minimum() => self.remap(op_id, y),
                        BOp::BitShiftLeft if cx.is_zero() => self.remap(op_id, y),
                        BOp::BitShiftRight if cx.is_zero() => self.remap(op_id, y),
                        _ => {}
                    },
                    (_, Op::Const(cy)) => match bop {
                        BOp::Add if cy.is_zero() => self.remap(op_id, x),
                        BOp::Sub if cy.is_zero() => self.remap(op_id, x),
                        BOp::Div if cy.is_zero() => panic!("Division by constant zero"),
                        BOp::Div if cy.is_one() => self.remap(op_id, x),
                        BOp::Div if cy.is_power_of_two() && cy.dtype() == IDX_T => {
                            let y = self.insert_before(op_id, Op::Const(cy.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::BitShiftRight };
                        }
                        BOp::Mod if cy.is_zero() => panic!("Module by constant zero"),
                        BOp::Mod if cy.is_zero() && cy.dtype() == IDX_T => {
                            let shift = Constant::binary(cy, Constant::idx(1), BOp::Sub);
                            let y = self.insert_before(op_id, Op::Const(shift));
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::BitAnd };
                        }
                        // Consecutive modulo by constant, pick smallest constant
                        BOp::Mod if cy.dtype() == IDX_T => {
                            if let Op::Binary { bop, x: xi, y: yi } = self.ops[x].op {
                                if bop == BOp::Mod
                                    && let Op::Const(ciy) = self.ops[yi].op
                                {
                                    if ciy > cy {
                                        self.ops[op_id].op = Op::Binary { x: xi, y, bop: BOp::Mod };
                                    } else {
                                        self.ops[op_id].op = Op::Binary { x: xi, y: yi, bop: BOp::Mod };
                                    }
                                }
                            }
                        }
                        BOp::Pow if cy.is_zero() => self.ops[op_id].op = Op::Const(cy.dtype().one_constant()),
                        BOp::Pow if cy.is_one() => self.remap(op_id, x),
                        BOp::Pow if cy.is_two() => self.ops[op_id].op = Op::Binary { x, y: x, bop: BOp::Mul },
                        BOp::BitShiftLeft if cy.is_zero() => self.remap(op_id, x),
                        BOp::BitShiftRight if cy.is_zero() => self.remap(op_id, x),
                        _ => {}
                    },
                    (x_op, y_op) if x_op == y_op => {
                        match bop {
                            BOp::Div => todo!(), // should be constant 1
                            BOp::Sub => todo!(), // should be constant 0
                            _ => {}
                        }
                    }
                    _ => {}
                },
                Op::Mad { x, y, z } => {
                    match (self.at(x).clone(), self.at(y).clone(), self.at(z).clone()) {
                        (Op::Const(cx), Op::Const(cy), Op::Const(cz)) => {
                            let mul = Constant::binary(cx, cy, BOp::Mul);
                            self.ops[op_id].op = Op::Const(Constant::binary(mul, cz, BOp::Add));
                        }
                        (Op::Const(cx), Op::Const(cy), _) => {
                            let mul = Constant::binary(cx, cy, BOp::Mul);
                            let x = self.insert_before(op_id, Op::Const(mul));
                            self.ops[op_id].op = Op::Binary { x, y: z, bop: BOp::Add };
                        }
                        (Op::Const(cx), _, _) if cx.is_zero() => {
                            self.remap(op_id, z);
                        }
                        (Op::Const(cx), _, _) if cx.is_one() => {
                            self.ops[op_id].op = Op::Binary { x: y, y: z, bop: BOp::Add };
                        }
                        (_, Op::Const(cy), _) if cy.is_zero() => {
                            self.remap(op_id, z);
                        }
                        (_, Op::Const(cy), _) if cy.is_one() => {
                            self.ops[op_id].op = Op::Binary { x, y: z, bop: BOp::Add };
                        }
                        (_, _, Op::Const(cz)) if cz.is_zero() => {
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::Mul };
                        }
                        _ => {}
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    // Eliminates accs that are not stored into in loops
    pub fn fold_accs(&mut self) {
        // We have to do constant folding before folding accs to guarantee indices are constants
        self.constant_folding(0);
        // Check if a define exists without a loop that stores into that define
        let mut defines = Map::default();
        let mut loop_level = 0u32;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Define { scope, .. } => {
                    if scope == Scope::Register {
                        defines.insert(op_id, loop_level);
                    }
                }
                Op::Store { dst, .. } => {
                    //println!("Store to {dst}, loop_level={loop_level}");
                    if let Some(level) = defines.get(&dst) {
                        if loop_level > *level {
                            defines.remove(&dst);
                        }
                    }
                }
                Op::Loop { .. } => {
                    loop_level += 1;
                }
                Op::EndLoop => {
                    loop_level -= 1;
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        //println!("defines: {defines:?}");
        for (define, _) in defines {
            self.fold_acc(define);
        }
    }

    pub fn fold_acc(&mut self, define_id: OpId) {
        //println!("Folding acc {define_id}");
        let Op::Define { len, .. } = self.ops[define_id].op else { unreachable!() };
        self.remove_op(define_id);
        let mut latest_stores = vec![OpId::NULL; len];

        let mut remaps = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match *self.at(op_id) {
                Op::Store { dst, x, index, vlen } => {
                    if vlen > 1 {
                        todo!()
                    }
                    if dst == define_id {
                        self.remove_op(op_id);
                        // x may have been removed as a previous load. If that was the case, the load was redundant
                        if self.ops.contains_key(x) {
                            let Op::Const(index) = self.ops[index].op else { unreachable!() };
                            let Constant::U32(index) = index else { unreachable!() };
                            latest_stores[index as usize] = x;
                            //println!("Latest stores = {latest_stores:?}");
                        }
                        op_id = next;
                        continue;
                    }
                }
                Op::Load { src, index, .. } => {
                    if src == define_id {
                        self.remove_op(op_id);
                        let Op::Const(index) = self.ops[index].op else { unreachable!() };
                        let Constant::U32(index) = index else { unreachable!() };
                        remaps.insert(op_id, latest_stores[index as usize]);
                        op_id = next;
                        continue;
                    }
                }
                _ => {}
            }
            self.ops[op_id].op.remap_params(&remaps);
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    // Loops that don't contain stores can be deleted
    pub fn delete_empty_loops(&mut self) {
        let mut stack: Vec<(bool, Vec<OpId>)> = Vec::new();
        let mut dead = Set::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            for s in &mut stack {
                s.1.push(op_id);
            }
            match self.at(op_id) {
                Op::Loop { .. } => stack.push((false, vec![op_id])),
                Op::Store { .. } => {
                    for s in &mut stack {
                        s.0 = true
                    }
                }
                Op::EndLoop => {
                    let (has_store, ops) = stack.pop().unwrap();
                    if has_store {
                        if let Some(p) = stack.last_mut() {
                            p.1.extend(ops);
                        }
                    } else {
                        dead.extend(ops);
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        for op_id in dead {
            self.remove_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        let mut visited = Set::default();
        // We go backward from Stores and gather all needed ops, but we can't remove Loop and Define ops
        for (op_id, op) in self.iter_unordered() {
            if matches!(
                op,
                Op::Store { .. }
                    | Op::WMMA { .. }
                    | Op::Loop { .. }
                    | Op::Define { .. }
                    | Op::EndLoop { .. }
                    | Op::StoreView { .. }
            ) {
                params.push(op_id);
            }
        }
        while let Some(op_id) = params.pop() {
            if visited.insert(op_id) {
                params.extend(self.at(op_id).parameters());
            }
        }
        //self.ops.retain(|op_id| visited.contains(op_id));
        for op_id in self.ops.ids().collect::<Vec<_>>() {
            if !visited.contains(&op_id) {
                self.remove_op(op_id);
            }
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn common_subexpression_elimination(&mut self) {
        let mut stack: Vec<Map<Op, OpId>> = Vec::with_capacity(10);
        stack.push(Map::with_capacity_and_hasher(50, Default::default()));

        let mut remaps = Map::with_capacity_and_hasher(10, Default::default());
        let mut op_id = self.head;
        while !op_id.is_null() {
            let temp = self.next_op(op_id);
            match &mut self.ops[op_id].op {
                Op::Define { .. } => {} // skip define ops, these can not be deduplicated
                Op::Loop { .. } => {
                    stack.push(Map::with_capacity_and_hasher(50, Default::default()));
                }
                Op::EndLoop => {
                    stack.pop();
                }
                op => {
                    let mut remove_op = false;
                    for loop_level in &stack {
                        if let Some(&old_op_id) = loop_level.get(op) {
                            remaps.insert(op_id, old_op_id);
                            remove_op = true;
                            break;
                        }
                    }
                    if remove_op {
                        self.remove_op(op_id);
                    } else {
                        for param in op.parameters_mut() {
                            if let Some(&new_id) = remaps.get(param) {
                                *param = new_id;
                            }
                        }
                        stack.last_mut().unwrap().insert(op.clone(), op_id);
                    }
                }
            }
            op_id = temp;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn move_constants_to_beginning(&mut self) {
        let mut start = self.head;
        while let Op::Define { .. } = self.at(start) {
            start = self.next_op(start);
        }

        let mut op_id = start;
        let mut start = self.prev_op(start);
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Const(_) = self.at(op_id) {
                self.move_op_after(op_id, start);
                start = op_id;
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }
}
