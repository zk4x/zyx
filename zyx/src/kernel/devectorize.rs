// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Devectorize: unfold lane bundles left by `split` back to scalar IR.
//!
//! One pass, running in the device epilogue right before codegen, with
//! two mechanisms under a joint fixpoint: [`Kernel::devectorize`] expands
//! each lane-space fold ([`Op::Reduce`] whose `reduce_axis` mentions
//! unroll nodes) into the fold chain over its lanes, and unfolds every
//! remaining [`Op::Unroll`] node (a vector of constants `0..len`) into
//! one scalar statement per element. Every transport backend only ever
//! sees scalar IR.
//!
//! The unfolding walk is the duplication walk the old duplicating
//! `coarsen` ran, with the lane constants (`0..len`) in place of its
//! index offsets and without its group splitting, storage growth, or
//! accumulator special-casing (the split already grew storages and
//! lane-parameterized accumulator indices, so the generic
//! operand-substitution arm covers every op).

use crate::{Map, Set, dtype::Constant, kernel::{Kernel, Op, OpId}};

impl Kernel {
    /// Expand every lane-space fold into the fold chain over its lanes,
    /// then unfold every remaining unroll node into scalar statements.
    /// Unused nodes die in DCE. Runs in the device epilogue, before
    /// codegen. Folds take priority every round: a fold expansion can
    /// leave lane-conditional ops behind (axes the fold does not
    /// consume), and the unroll mechanism unfolds those next rounds.
    ///
    /// The expansion creates fresh copies for each fold lane; nothing
    /// runs after this joint with per-phase cleanups, so the folded
    /// duplicates are cleaned once at the end: constant folding turns
    /// per-lane index expressions into literals, CSE merges the
    /// identical forms, DCE removes what is dead.
    pub fn devectorize(&mut self) {
        for _ in 0..10000 {
            let found = self
                .iter_unordered()
                .find_map(|(op, x)| matches!(x, Op::Reduce { .. }).then_some(op));
            if let Some(reduce_id) = found {
                self.devectorize_reduce_one(reduce_id);
                continue;
            }
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

    /// Expand one lane-space fold: substitute the fold axis' unroll nodes
    /// with each lane's constant everywhere they are consumed, fold the
    /// per-lane values with the fold op, and remove the fold node.
    ///
    /// The axis expression may compose several unroll nodes (e.g.
    /// `unroll(axis0) * 4 + unroll(axis1)`); the chain then folds their
    /// product lexicographically (axis-ascending, first node most
    /// significant). Lane 0 rewrites the original ops in place; the
    /// remaining lanes insert as clones. Lane axes the fold does NOT
    /// consume are left untouched, so a multi-axis bundle folds along
    /// the fold axes only and the result keeps its other lanes — the
    /// fold-chain members stay lane-conditional on those surviving
    /// unroll nodes for the unroll rounds to unfold.
    fn devectorize_reduce_one(&mut self, reduce_id: OpId) {
        let Op::Reduce { x, rop, reduce_axis } = self.ops[reduce_id].op.clone() else {
            unreachable!("devectorize: not a reduce")
        };

        // The fold spaces: every unroll node mentioned by the axis
        // expression, axis-ascending. The axis cone may only compose
        // unroll nodes with binary arithmetic and constants (e.g.
        // `unroll(axis0) * 4 + unroll(axis1)`); anything else is not a
        // lane space this pass folds.
        let mut ordered: Vec<(OpId, u16, u32)> = Vec::new();
        {
            let mut stack = vec![reduce_axis];
            let mut seen: Set<OpId> = Set::default();
            for _ in 0..10_000 {
                let Some(id) = stack.pop() else { break };
                if !seen.insert(id) {
                    continue;
                }
                match self.at(id) {
                    Op::Unroll { axis, len } => ordered.push((id, *len, *axis)),
                    Op::Binary { x, y, .. } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    Op::Const(_) => {}
                    _ => todo!("devectorize: lane fold over a non-unroll axis"),
                }
            }
        }
        if ordered.is_empty() {
            todo!("devectorize: fold over no lane space");
        }
        ordered.sort_by_key(|&(_, _, axis)| axis);
        let fold_nodes: Vec<(OpId, u16)> = ordered.into_iter().map(|(id, len, _)| (id, len)).collect();
        let mut strides: Vec<u64> = Vec::with_capacity(fold_nodes.len());
        let fold_len: u64 = fold_nodes.iter().map(|&(_, l)| u64::from(l)).product();
        for i in 0..fold_nodes.len() {
            let mut st = 1u64;
            for &(_, l) in &fold_nodes[i + 1..] {
                st *= u64::from(l);
            }
            strides.push(st);
        }

        // Per-node lane constants; flat index `i` maps to node k's lane
        // `(i / stride_k) % len_k`, lane 0 first (the originals are
        // rewritten in place with lane 0).
        let mut lane_consts: Vec<Vec<OpId>> = Vec::with_capacity(fold_nodes.len());
        for &(node_id, len) in &fold_nodes {
            let mut consts = Vec::with_capacity(len as usize);
            for lane in 0..len {
                consts.push(self.insert_before(node_id, Op::Const(Constant::idx(lane))));
            }
            lane_consts.push(consts);
        }
        let lane_of = |flat: u64, k: usize| -> OpId {
            let len = lane_consts[k].len() as u64;
            let idx = (flat / strides[k]) % len;
            lane_consts[k][idx as usize]
        };

        // For remapping parameters: the per-op clones for flat lanes
        // 1..fold_len (lane 0 rewrites the originals in place).
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();

        // Duplicate every fold-conditional op once per remaining fold
        // lane: ops directly naming the fold's unroll nodes, and ops
        // consuming earlier-rewritten (remapped) ones. The fold's own
        // x-chain and everything downstream of it are covered by this
        // generic arm; the unroll nodes themselves are substituted away.
        // The walk is linear in IR order so a producer's clones are
        // registered in `remaps` before its consumers are checked; the
        // clones insert after the current op, so the pointer `next`
        // (its original successor) never re-visits them.
        let mut walk_id = self.head;
        while !walk_id.is_null() {
            let next = self.next_op(walk_id);
            if walk_id == reduce_id {
                // The fold being expanded never folds through itself;
                // other folds in its cone would need their own expansion
                // first.
                walk_id = next;
                continue;
            }
            match self.ops[walk_id].op.clone() {
                Op::Storage { .. } | Op::Range { .. } | Op::Unroll { .. } | Op::Loop { .. } | Op::EndLoop
                | Op::Barrier | Op::Param { .. } | Op::Const(_) => {}
                Op::Reduce { .. } => todo!("devectorize: fold cone containing another fold {walk_id}"),
                ref op => {
                    let mut lane_cond = false;
                    for param in op.parameters() {
                        if fold_nodes.iter().any(|&(f, _)| f == param) || remaps.contains_key(&param) {
                            lane_cond = true;
                            break;
                        }
                    }
                    if !lane_cond {
                        walk_id = next;
                        continue;
                    }
                    let mut ids = Vec::with_capacity(fold_len as usize - 1);
                    let mut at_id = walk_id;
                    for flat in 1..fold_len {
                        let mut lane_op = op.clone();
                        // Reindex the op.
                        for param in lane_op.parameters_mut() {
                            if let Some(ix) = fold_nodes.iter().position(|&(f, _)| f == *param) {
                                *param = lane_of(flat, ix);
                            } else if let Some(remap) = remaps.get(param) {
                                *param = remap[flat as usize - 1];
                            }
                        }
                        at_id = self.insert_after(at_id, lane_op);
                        ids.push(at_id);
                    }
                    // Lane 0 rewrites the original in place.
                    for param in self.ops[walk_id].op.parameters_mut() {
                        if let Some(ix) = fold_nodes.iter().position(|&(f, _)| f == *param) {
                            *param = lane_of(0, ix);
                        }
                    }
                    // Store which remaps will be used in the future.
                    remaps.insert(walk_id, ids);
                }
            }
            walk_id = next;
        }

        // The fold chain over per-lane values: the original (lane-0)
        // partial folds with every later lane's clone of the value.
        let mut acc = x;
        for flat in 1..fold_len {
            let lane_val = match remaps.get(&x) {
                Some(ids) => ids[flat as usize - 1],
                None => x,
            };
            acc = self.insert_before(reduce_id, Op::Binary { x: acc, y: lane_val, bop: rop });
        }

        // Downstream consumers of the fold now read the chain tail.
        let op_ids: Vec<OpId> = self.ops.ids().collect();
        for walk_id in op_ids {
            for param in self.ops[walk_id].op.parameters_mut() {
                if *param == reduce_id {
                    *param = acc;
                }
            }
        }
        self.remove_op(reduce_id);
    }

    /// Unfold a single unroll node: recreate every element-conditional op
    /// once per element with the node substituted by that element's
    /// constant, rewrite the originals in place as element 0, then remove
    /// the node. Returns false when the node must stay (none today).
    /// A `Reduce` here is a driver bug: the joint loop expands folds
    /// first, so none survive to the unroll rounds.
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
        kernel::{Dev, MemScope, autotune::Optimization, split},
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
        k.devectorize();

        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. })));
        k.verify();
    }
}
