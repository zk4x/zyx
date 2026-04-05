// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
    Map, Set,
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

        //println!("Upcast {op_id} by factor={factor}");
        //self.debug_colorless();

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

        /*if self
            .ops
            .iter()
            .any(|(id, node)| matches!(node.op, Op::Loop { .. }) && self.loop_uses_gidx(id, op_id))
        {
            return;
        }*/

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
        let mut remap: Map<OpId, OpId> = Map::default();
        let latest_mad;
        loop {
            let next = self.next_op(id);
            if !matches!(
                self.ops[id].op,
                Op::Const(_) | Op::Index { .. } | Op::Define { scope: Scope::Global | Scope::Local, .. }
            ) {
                factor_const = self.insert_before(id, Op::Const(Constant::idx(factor as u64)));
                upcast_loop = self.insert_before(id, Op::Loop { len: factor });
                latest_mad = self.insert_after(upcast_loop, Op::Mad { x: op_id, y: factor_const, z: upcast_loop });
                remap.insert(op_id, latest_mad);
                break;
            }
            id = next;
        }

        let mut loop_depth = 0;
        let mut seen_in_scope: Set<OpId> = Set::default();
        let mut latest_mad = OpId::NULL;
        let mut acc_defines: Set<OpId> = Set::default();

        while !id.is_null() {
            let next = self.next_op(id);
            match self.ops[id].op {
                Op::Loop { .. } => {
                    if loop_depth == 0 {
                        self.insert_before(id, Op::EndLoop);
                        upcast_loop = self.insert_after(id, Op::Loop { len: factor });
                        latest_mad = self.insert_after(upcast_loop, Op::Mad { x: op_id, y: factor_const, z: upcast_loop });
                        remap.insert(op_id, latest_mad);
                        seen_in_scope.clear();
                    }
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        self.insert_before(id, Op::EndLoop);
                        upcast_loop = self.insert_after(id, Op::Loop { len: factor });
                        latest_mad = self.insert_after(upcast_loop, Op::Mad { x: op_id, y: factor_const, z: upcast_loop });
                        remap.insert(op_id, latest_mad);
                        seen_in_scope.clear();
                    }
                }
                Op::Define { .. } if loop_depth == 0 => {
                    seen_in_scope.insert(id);
                    self.move_op_before(id, upcast_loop);
                    if let Op::Define { len, .. } = &mut self.ops[id].op {
                        *len = *len * factor;
                    }
                    acc_defines.insert(id);
                }
                _ => {
                    seen_in_scope.insert(id);
                    let params: Vec<OpId> = self.ops[id].op.parameters().collect();
                    let mut new_params = Vec::new();
                    for param in params {
                        if let Some(&replacement) = remap.get(&param) {
                            new_params.push(replacement);
                        } else if !seen_in_scope.contains(&param) {
                            let cloned = self.clone_dep(param, latest_mad, &mut remap, &mut seen_in_scope, factor_const);
                            new_params.push(cloned);
                        } else {
                            new_params.push(param);
                        }
                    }
                    // Now remap using the collected new params
                    let mut param_iter = new_params.into_iter();
                    for param in self.ops[id].op.parameters_mut() {
                        if let Some(new_param) = param_iter.next() {
                            *param = new_param;
                        }
                    }
                    // Fix accumulator indexing for Load/Store
                    if let Op::Load { src, index, .. } = &self.ops[id].op {
                        if acc_defines.contains(src) {
                            let mad = self.insert_before(id, Op::Mad { x: *index, y: factor_const, z: upcast_loop });
                            if let Op::Load { index: load_index, .. } = &mut self.ops[id].op {
                                *load_index = mad;
                            }
                        }
                    }
                    if let Op::Store { dst, index, .. } = &self.ops[id].op {
                        if acc_defines.contains(dst) {
                            let mad = self.insert_before(id, Op::Mad { x: *index, y: factor_const, z: upcast_loop });
                            if let Op::Store { index: store_index, .. } = &mut self.ops[id].op {
                                *store_index = mad;
                            }
                        }
                    }
                }
            }
            id = next;
        }

        // Ensure kernel ends with EndLoop
        self.insert_after(self.tail, Op::EndLoop);

        //println!(" ==== After upcast: ==== ");
        //self.debug_colorless();

        self.verify();
    }

    fn loop_uses_gidx(&self, loop_id: OpId, gidx_id: OpId) -> bool {
        println!("Checking for usage of loop={loop_id} gidx={gidx_id}");
        let mut id = self.next_op(loop_id);
        let mut depth = 1;
        while !id.is_null() {
            match self.ops[id].op {
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => {
                    depth -= 1;
                    if depth == 0 {
                        println!("not used, end loop");
                        return false;
                    }
                }
                _ => {
                    let mut stack = vec![id];
                    let mut uses_loop = false;
                    let mut uses_gidx = false;
                    let mut visited = Set::default();
                    while let Some(cur) = stack.pop() {
                        if cur == loop_id {
                            uses_loop = true;
                        }
                        if cur == gidx_id {
                            uses_gidx = true;
                        }
                        if uses_loop && uses_gidx {
                            println!("used");
                            return true;
                        }
                        if !visited.insert(cur) {
                            continue;
                        }
                        if !self.ops.contains_key(cur) {
                            continue;
                        }
                        for param in self.ops[cur].op.parameters() {
                            stack.push(param);
                        }
                    }
                }
            }
            id = self.next_op(id);
        }
        false
    }

    fn clone_dep(
        &mut self,
        dep_id: OpId,
        after: OpId,
        remap: &mut Map<OpId, OpId>,
        seen: &mut Set<OpId>,
        factor_const: OpId,
    ) -> OpId {
        if let Some(&replacement) = remap.get(&dep_id) {
            return replacement;
        }
        if seen.contains(&dep_id) {
            return dep_id;
        }

        // Don't clone Index, Const, Define, Loop, EndLoop - they're always accessible
        if matches!(
            self.ops[dep_id].op,
            Op::Index { .. } | Op::Const(_) | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop
        ) {
            return dep_id;
        }

        // Recursively clone dependencies first
        let op = self.ops[dep_id].op.clone();
        let mut cloned = op;
        for param in cloned.parameters_mut() {
            if let Some(&replacement) = remap.get(param) {
                *param = replacement;
            } else if !seen.contains(param) {
                *param = self.clone_dep(*param, after, remap, seen, factor_const);
            }
        }

        let new_id = self.insert_after(after, cloned);
        remap.insert(dep_id, new_id);
        seen.insert(new_id);
        new_id
    }
}
