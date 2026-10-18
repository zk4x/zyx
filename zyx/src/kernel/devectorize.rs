// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Devectorize: unfold unroll vectors left by `split` back to scalar IR.
//!
//! One pass, running in the device epilogue right before codegen:
//! [`Kernel::devectorize_unrolls`] duplicates every op still conditional
//! on a remaining [`Op::Unroll`] node (a vector of constants `0..len`),
//! substituting the node with each element's constant.
//! Every transport backend only ever sees scalar IR.

use crate::{Map, dtype::Constant, kernel::{Kernel, Op, OpId}};

impl Kernel {
    /// Unfold every used `Op::Unroll` node into one scalar statement
    /// per element. Unused nodes are removed by DCE. Runs in the device
    /// epilogue, before codegen.
    ///
    /// The unfolding creates fresh copies for each element; the joint tail
    /// cleans them: constant folding turns per-element index expressions
    /// into literals, CSE merges the identical forms, DCE removes dead.
    pub fn devectorize_unrolls(&mut self) {
        for _ in 0..10000 {
            let found =
                self.iter_unordered().find_map(|(op, x)| matches!(x, Op::Unroll { .. }).then_some(op));
            let Some(unroll_id) = found else { break };
            // A refused node (none exist today) stays in place — stop
            // instead of looping on it forever.
            if !self.devectorize_unroll_one(unroll_id) {
                break;
            }
        }
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.verify();
    }

    /// Unfold a single unroll node: recreate every element-conditional op
    /// once per element with the node substituted by that element's
    /// constant, rewrite the originals in place as element 0, then remove
    /// the node. Returns false when the node must stay (none today).
    fn devectorize_unroll_one(&mut self, unroll_id: OpId) -> bool {
        let len = match self.at(unroll_id) {
            Op::Unroll { len, .. } => u64::from(*len),
            _ => unreachable!("devectorize: not an unroll node"),
        };

        // Element constants, element 0 first (element 0 rewrites the originals).
        let elem0 = self.insert_before(unroll_id, Op::Const(Constant::idx(0u64)));
        let mut elems = Vec::with_capacity((len - 1) as usize);
        for i in 1..len {
            elems.push(self.insert_before(unroll_id, Op::Const(Constant::idx(i))));
        }

        // First skip ops that don't need duplication.
        let mut walk_id = self.head;
        while !walk_id.is_null()
            && matches!(
                self.ops[walk_id].op,
                Op::Storage { .. } | Op::Param { .. } | Op::Range { .. } | Op::Const(_) | Op::Unroll { .. }
            )
        {
            walk_id = self.next_op(walk_id);
        }

        // For remapping parameters.
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();

        // Now loop over remaining ops and duplicate element-conditional ones.
        // The walk is linear in IR order so a producer's clones are
        // registered in `remaps` before its consumers are checked; the
        // clones insert after the current op, so the pointer `next` (its
        // original successor) never re-visits them.
        while !walk_id.is_null() {
            let next = self.next_op(walk_id);
            match self.ops[walk_id].op.clone() {
                Op::Storage { .. } | Op::Range { .. } | Op::Unroll { .. } | Op::Loop { .. } | Op::EndLoop
                | Op::Barrier | Op::Param { .. } | Op::Const(_) => {}
                Op::Reduce { .. } => unreachable!("fold not devectorized before the unroll pass"),
                ref op => {
                    let mut elem_cond = false;
                    for param in op.parameters() {
                        if param == unroll_id || remaps.contains_key(&param) {
                            elem_cond = true;
                            break;
                        }
                    }
                    if !elem_cond {
                        walk_id = next;
                        continue;
                    }
                    let mut ids = Vec::with_capacity((len - 1) as usize);
                    let mut at_id = walk_id;
                    for i in 0..(len - 1) as usize {
                        let mut elem_op = op.clone();
                        // Reindex the op.
                        for param in elem_op.parameters_mut() {
                            if *param == unroll_id {
                                *param = elems[i];
                            } else if let Some(remap) = remaps.get(param) {
                                *param = remap[i];
                            }
                        }
                        at_id = self.insert_after(at_id, elem_op);
                        ids.push(at_id);
                    }
                    // Element 0 rewrites the original in place (the unroll
                    // node has no inherent element-0 meaning to inherit).
                    for param in self.ops[walk_id].op.parameters_mut() {
                        if *param == unroll_id {
                            *param = elem0;
                        }
                    }
                    // Store which remaps will be used in the future.
                    remaps.insert(walk_id, ids);
                }
            }
            walk_id = next;
        }

        self.remove_op(unroll_id);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        backend::DeviceInfo,
        kernel::{BOp, Dev, MemScope, autotune::Optimization, split},
        shape::Dim,
    };
    use std::sync::Arc;

    fn cpu() -> DeviceInfo {
        DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [crate::backend::DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }
    }

    /// A reduce loop split at factor 4 duplicates the body per element and
    /// chains the per-element values into the acc store: no `Op::Reduce`,
    /// no `Op::Unroll`, the accumulator keeps length 1.
    /// A split loop unrolls into duplicated scalar IR: after the pass no
    /// `Op::Unroll` node and no `Op::Reduce` survive.
    #[test]
    fn devectorize_unrolls_expands_scalars() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let init = k.const_val(0.0f32);
        k.store(acc, init, 0i64);
        let loop_id = {
            let mut lv = OpId::NULL;
            let acc_ref = acc;
            k.loop_over(64i64, |k, lv_id| {
                lv = lv_id;
                let x = k.load(inp, lv_id);
                let prev = k.load(acc_ref, 0i64);
                let sum = k.add(prev, x);
                k.store(acc_ref, sum, 0i64);
            });
            lv
        };
        let result = k.load(acc, 0i64);
        k.store(out, result, 0i64);

        let split = split::OptSplit { configs: vec![(loop_id,
            split::SplitMode::Unroll, 4)] };
        Optimization::apply(&split, &mut k, 0);
        k.devectorize_unrolls();

        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. })));
        k.verify();
    }
}
