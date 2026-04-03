// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Set,
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

        // Step 1: Find reduce loop
        let reduce_loop = self.next_loop_after(op_id);

        eprintln!("DEBUG: op_id={:?}, reduce_loop={:?}", op_id, reduce_loop);

        if reduce_loop == OpId::NULL {
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

        // Step 2: Find accumulator define
        let acc_def = self.find_accumulator_before(op_id, reduce_loop);
        eprintln!("DEBUG: acc_def={:?}", acc_def);
        if acc_def == OpId::NULL {
            return;
        }

        // Step 3: Find reduce loop end
        let reduce_end = self.find_matching_end(self.next_op(reduce_loop));
        eprintln!("DEBUG: reduce_end={:?}", reduce_end);
        if reduce_end == OpId::NULL {
            return;
        }

        // Found: acc_def, reduce_loop, reduce_end - just verify they exist
        let pre_loop_deps = self.pre_loop_deps(reduce_loop, reduce_end);
        eprintln!("DEBUG: pre_loop_deps={:?}", pre_loop_deps);

        println!(" ==== After upcast: ==== ");
        self.debug_colorless();

        self.verify();
    }

    fn pre_loop_deps(&self, reduce_loop: OpId, reduce_end: OpId) -> Set<OpId> {
        let mut inside = Set::default();
        let mut result = Set::default();
        let mut visited = Set::default();

        let mut id = self.next_op(reduce_loop);
        while id != reduce_end {
            inside.insert(id);
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

    fn next_loop_after(&self, after: OpId) -> OpId {
        let mut id = self.next_op(after);
        while !id.is_null() {
            if matches!(self.ops[id].op, Op::Loop { .. }) {
                return id;
            }
            id = self.next_op(id);
        }
        OpId::NULL
    }

    fn find_accumulator_before(&self, after: OpId, before: OpId) -> OpId {
        let mut id = self.next_op(after);
        while id != before {
            if let Op::Define { .. } = self.ops[id].op {
                return id;
            }
            id = self.next_op(id);
        }
        OpId::NULL
    }

    fn find_matching_end(&self, start: OpId) -> OpId {
        let mut depth = 1;
        let mut id = start;
        while !id.is_null() {
            match self.ops[id].op {
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => {
                    depth -= 1;
                    if depth == 0 {
                        return id;
                    }
                }
                _ => {}
            }
            id = self.next_op(id);
        }
        OpId::NULL
    }
}
