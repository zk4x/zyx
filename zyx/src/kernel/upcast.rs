// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn opt_upcast(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global {
                    for &f in &[4, 8, 2, 16] {
                        if len.is_multiple_of(f) && len / f >= 4 {
                            factors.push((op_id, f));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::Upcast { factors }, n_configs)
    }

    pub fn upcast(&mut self, op_id: OpId, factor: usize) {
        // Verify op_id is a global index
        debug_assert!(matches!(self.ops[op_id].op, Op::Index { scope: Scope::Global, .. }));

        println!("Upcast {op_id} by factor={factor}");

        if !self.ops.values().any(|node| matches!(node.op, Op::Loop { .. })) {
            // No reduce loop - just split the dimension
            let Op::Index { len, scope, axis } = self.ops[op_id].op else {
                return;
            };
            self.split_dim(
                op_id,
                vec![
                    Op::Index { len: len / factor, scope, axis },
                    Op::Index { len: factor, scope: Scope::Local, axis },
                ],
            );
            return;
        }

        // === UPCAST WITH REDUCE LOOPS === //

        let Op::Index { len, .. } = &mut self.ops[op_id].op else {
            unreachable!()
        };
        debug_assert!(len.is_multiple_of(factor));
        *len /= factor;

        // Single pass over all kernel ops

        // Find first non-trivial op
        let factor_const;
        let mut upcast_loop;
        let mut id = self.head;
        loop {
            let next = self.next_op(id);
            if !matches!(
                self.ops[id].op,
                Op::Const(_) | Op::Index { .. } | Op::Define { scope: Scope::Global | Scope::Local, .. }
            ) {
                factor_const = self.insert_before(id, Op::Const(Constant::idx(factor as u64)));
                upcast_loop = self.insert_before(id, Op::Loop { len: factor });
                break;
            }
            id = next;
        }

        // Fix reduce loops and their dependencies
        // Resize accumulators
        // Fix indexing
        // THIS IS THE LOOP THAT DOES EVERYTHING

        let mut loop_depth = 0;
        let mut remap: Map<OpId, OpId> = Map::default();

        while !id.is_null() {
            let next = self.next_op(id);
            match self.ops[id].op {
                Op::Loop { .. } => {
                    if loop_depth == 0 {
                        self.insert_before(id, Op::EndLoop);
                        upcast_loop = self.insert_after(id, Op::Loop { len: factor });
                        let mul = self.insert_after(upcast_loop, Op::Binary { x: op_id, y: factor_const, bop: BOp::Mul });
                        let mad = self.insert_after(mul, Op::Binary { x: mul, y: upcast_loop, bop: BOp::Add });
                        remap.insert(op_id, mad);
                    }
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        self.insert_before(id, Op::EndLoop);
                        upcast_loop = self.insert_after(id, Op::Loop { len: factor });
                        let mul = self.insert_after(upcast_loop, Op::Binary { x: op_id, y: factor_const, bop: BOp::Mul });
                        let mad = self.insert_after(mul, Op::Binary { x: mul, y: upcast_loop, bop: BOp::Add });
                        remap.insert(op_id, mad);
                    }
                }
                _ => {
                    for param in self.ops[id].op.parameters_mut() {
                        if let Some(&replacement) = remap.get(param) {
                            *param = replacement;
                        }
                    }
                }
            }
            id = next;
        }

        // Ensure kernel ends with EndLoop
        self.insert_after(self.tail, Op::EndLoop);

        println!(" ==== After upcast: ==== ");
        self.debug_colorless();

        self.verify();
    }

    fn pre_loop_deps(&self, reduce_loop: OpId) -> Set<OpId> {
        let mut inside = Set::default();
        let mut result = Set::default();
        let mut visited = Set::default();

        let mut id = reduce_loop;
        let mut loop_depth = 0;
        while !id.is_null() {
            inside.insert(id);
            match self.ops[id].op {
                Op::Loop { .. } => loop_depth += 1,
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            let mut stack: Vec<OpId> = self.ops[id].op.parameters().collect();
            while let Some(param) = stack.pop() {
                if visited.contains(&param) || inside.contains(&param) {
                    continue;
                }
                visited.insert(param);
                if matches!(self.ops[param].op, Op::Const(_) | Op::Define { .. } | Op::Index { .. }) {
                    continue;
                }
                result.insert(param);
                stack.extend(self.ops[param].op.parameters());
            }
            id = self.next_op(id);
        }

        result
    }
}
