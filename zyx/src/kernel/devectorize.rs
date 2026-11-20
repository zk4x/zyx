// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Devectorize: unfold lane bundles left by `split` back to scalar IR.
//!
//! One pass, running in the device epilogue right before codegen. A
//! lane-space reduce ([`Op::Reduce`] whose `reduce_axis` mentions unroll
//! nodes) is expanded first into the reduce chain over its lanes, because it
//! collapses the lane instead of repeating it. Then every remaining
//! [`Op::Unroll`] node is decided once: it stays a vector width where the
//! accesses it shapes are contiguous runs the device renders, and the
//! dependents that cannot carry it that way are unfolded into one scalar
//! statement per element. Every transport backend only ever sees scalar or
//! vector IR, never a work bundle.
//!
//! The unfolding walk is the duplication walk the old duplicating
//! `coarsen` ran, with the lane constants (`0..len`) in place of its
//! index offsets and without its group splitting, storage growth, or
//! accumulator special-casing (the split already grew storages and
//! lane-parameterized accumulator indices, so the generic
//! operand-substitution arm covers every op).

use crate::{
    DType, Map, Set, ZyxError,
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, PTXOp},
    shape::Dim,
};

/// Which expansion is duplicating lanes, and what it needs on top of the
/// generic substitution.
#[derive(Clone, Copy)]
enum DupMode<'a> {
    /// One split lane: only the decided `dup` set unfolds, a lane extract is
    /// repointed at the lane's own copy, and a fragment operand packs its
    /// per-lane pieces into one instruction.
    Unroll { dup: &'a Set<OpId> },
    /// Every lane of a reduce's axis space; the caller folds the result.
    Reduce { reduce_id: OpId },
}

impl Kernel {
    /// Expand every lane-space reduce into the reduce chain over its lanes,
    /// then unfold every remaining unroll node into scalar statements.
    /// Unused nodes die in DCE. Runs in the device epilogue, before
    /// codegen. Reduces take priority every round: a reduce expansion can
    /// leave lane-conditional ops behind (axes the reduce does not
    /// consume), and the unroll mechanism unfolds those next rounds.
    ///
    /// The expansion creates fresh copies for each reduce lane; nothing runs
    /// after it with per-phase cleanups, so the duplicates are cleaned once at
    /// the end: constant folding turns per-lane index expressions into
    /// literals, CSE merges the identical forms, DCE removes what is dead.
    pub fn devectorize(&mut self) -> Result<(), ZyxError> {
        if std::env::var_os("ZYX_DV").is_some() {
            self.debug();
        }
        // A lane reduce collapses its lane; it is not a repeat.
        loop {
            self.check_op_limit()?;
            let mut found = OpId::NULL;
            let mut walk_id = self.head;
            while !walk_id.is_null() {
                if matches!(self.at(walk_id), Op::Reduce { .. }) {
                    found = walk_id;
                    break;
                }
                walk_id = self.next_op(walk_id);
            }
            if found.is_null() {
                break;
            }
            self.devectorize_reduce_one(found);
        }
        // One decision for every lane, then one pass per lane. The decision is
        // recomputed per lane: expanding an earlier lane inserts clones that
        // consume the later lanes too, so a decision taken before them would
        // miss the clones. The lane ids themselves are stable — the unfolding
        // never inserts an `Unroll`, and only a lane's own round removes it.
        let lanes: Vec<OpId> =
            self.iter_unordered().filter_map(|(id, op)| matches!(op, Op::Unroll { .. }).then_some(id)).collect();
        for lane in lanes {
            self.check_op_limit()?;
            let dup = self.lane_expansion(lane);
            if dup.is_empty() {
                continue;
            }
            self.devectorize_unroll_one(lane, &dup);
        }
        self.constant_folding();
        self.forward_register_stores();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        if std::env::var_os("ZYX_DV_AFTER").is_some() {
            self.debug();
        }
        self.verify();
        Ok(())
    }

    /// The render ceiling: the expansion multiplies the IR by the lane
    /// count, and the constant folding after it is superlinear.
    fn check_op_limit(&self) -> Result<(), ZyxError> {
        let n_ops = self.ops.len().0;
        if n_ops > crate::kernel::MAX_OPS {
            return Err(ZyxError::kernel_error(
                format!("devectorize: kernel IR has {n_ops} ops, above the {} -op limit", crate::kernel::MAX_OPS).into(),
            ));
        }
        Ok(())
    }

    /// Whether `Index(vec, k)` names lane `k` of `vec`: the value's lane is
    /// the innermost unit-stride term of its address, so one lane is one
    /// element. An instruction result or a `Stack` has component dims, and a
    /// strided lane has pieces; neither is a lane extract.
    fn lane_extract(&self, vec: OpId, lane: OpId) -> bool {
        match self.at(vec) {
            Op::Load { src_idx, .. } | Op::Copy { src_idx, .. } => !src_idx.is_null() && self.unit_lane_term(*src_idx, lane),
            _ => false,
        }
    }

    /// The slot run an index op names: `(base, count)`.
    ///
    /// After `devectorize` a register-storage index is either a literal or
    /// `base + lane`, where the lane is an `Unroll` kept because the access it
    /// shapes is a vector width — the store and the value it writes then cover
    /// the same contiguous run. Anything else (`base + thread`, a compound
    /// expression) is not a slot the pass can name, and is left alone.
    fn slot_run(&self, idx: OpId) -> Option<(i64, i64)> {
        if let Some(base) = self.resolve_const(idx).and_then(Constant::as_dim) {
            return Some((base, 1));
        }
        // A lane on its own is the run `0..len`.
        if let Op::Unroll { len, .. } = self.at(idx) {
            return Some((0, i64::from(*len)));
        }
        let Op::Binary { x, y, bop: BOp::Add } = self.at(idx) else {
            return None;
        };
        for (a, b) in [(*x, *y), (*y, *x)] {
            if let Op::Unroll { len, .. } = self.at(a) {
                if let Some(base) = self.resolve_const(b).and_then(Constant::as_dim) {
                    return Some((base, i64::from(*len)));
                }
            }
        }
        None
    }

    /// Forward `MemScope::Register` storage traffic to the values stored in it.
    ///
    /// A register storage is not memory: once the lanes are unfolded its slots
    /// are literal indices, so `store s[i] = v` followed by `load s[i + k]` is a
    /// register rename. Forwarding the load to the slice of `v` that it names
    /// takes the storage out of the dataflow, so a consumer depends on its
    /// producer directly: a `wmma` reads the `ldmatrix` that staged it instead
    /// of reading it back through an array, which is what lets the scheduler
    /// count the fragment loads it has in flight and frees the registers the
    /// staging array held.
    ///
    /// Only straight-line traffic is forwarded: a loop edge, a barrier, an
    /// opaque side effect or a copy clears what is known, so a load can never
    /// pick up a value from another iteration or from the other side of a sync.
    fn forward_register_stores(&mut self) {
        // storage slot -> (value, offset inside that value) for the slots whose
        // contents are known at this point of the walk.
        let mut origin: Map<(OpId, i64), (OpId, usize)> = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.at(op_id) {
                Op::Loop { .. } | Op::EndLoop | Op::Barrier | Op::Copy { .. } | Op::TT(crate::kernel::TTOp::Copy { .. }) => {
                    origin.clear()
                }
                Op::PTX(p) if matches!(**p, PTXOp::WarpSync) => origin.clear(),
                Op::Store { dst, dst_idx, src } => {
                    let dst = *dst;
                    let src = *src;
                    let is_reg = matches!(self.at(dst), Op::Storage { scope: MemScope::Register, .. });
                    // The index's run length *is* the access width, so the value
                    // covers exactly those slots. A broadcast value (one element
                    // stored into a run of slots) has no per-slot mapping: the
                    // load side would index a scalar, so it stays in the storage.
                    if let (true, Some((base, len))) = (is_reg, self.slot_run(*dst_idx)) {
                        // The mapping is only well defined when the value is
                        // exactly the run: a broadcast scalar or a wider value
                        // has no per-slot mapping, and forwarding it would
                        // index an element the run does not hold.
                        if self.elements(src) == Dim::from(len) {
                            for k in 0..len {
                                origin.insert((dst, base + k), (src, k as usize));
                            }
                        }
                    }
                }
                Op::Load { src, src_idx } => {
                    let src = *src;
                    let is_reg = matches!(self.at(src), Op::Storage { scope: MemScope::Register, .. });
                    let (true, Some((base, len))) = (is_reg, self.slot_run(*src_idx)) else {
                        op_id = next;
                        continue;
                    };
                    // Each element is forwarded to the value that stored it: a
                    // fragment can span two staging results, so the elements are
                    // gathered one by one and only the single-value case uses the
                    // stored value directly.
                    let mut from: Vec<(OpId, usize)> = Vec::with_capacity(len as usize);
                    for k in 0..len {
                        match origin.get(&(src, base + k)) {
                            Some(&(v, off)) => from.push((v, off)),
                            None => {
                                from.clear();
                                break;
                            }
                        }
                    }
                    if from.is_empty() {
                        op_id = next;
                        continue;
                    }
                    let (first, start) = from[0];
                    let contiguous = from.iter().enumerate().all(|(k, &(v, off))| v == first && off == start + k);
                    let scalar = self.shape(first).iter().product::<i64>() == 1 && from.iter().all(|&(v, _)| v == first);
                    let whole = contiguous && start == 0 && self.shape(first).iter().product::<i64>() == len;
                    let replacement = if whole || scalar {
                        // A broadcast scalar names every slot it was stored in;
                        // an `Index` would need a vector to index.
                        first
                    } else if len == 1 {
                        self.insert_before(op_id, Op::Index { vec: first, idx: start })
                    } else {
                        let mut ops = Vec::with_capacity(from.len());
                        for &(v, off) in &from {
                            ops.push(self.insert_before(op_id, Op::Index { vec: v, idx: off }));
                        }
                        self.insert_before(op_id, Op::Stack { ops: ops.into() })
                    };
                    self.remap(op_id, replacement);
                }
                _ => {}
            }
            op_id = next;
        }
    }

    /// Expand one lane-space reduce: substitute the reduce axis' unroll nodes
    /// with each lane's constant everywhere they are consumed, reduce the
    /// per-lane values with the reduce op, and remove the reduce node.
    ///
    /// The axis expression may compose several unroll nodes (e.g.
    /// `unroll(axis0) * 4 + unroll(axis1)`); the chain then reduces their
    /// product lexicographically (axis-ascending, first node most
    /// significant). Lane 0 rewrites the original ops in place; the
    /// remaining lanes insert as clones. Lane axes the reduce does NOT
    /// consume are left untouched, so a multi-axis bundle reduces along
    /// the reduce axes only and the result keeps its other lanes — the
    /// reduce-chain members stay lane-conditional on those surviving
    /// unroll nodes for the unroll rounds to unfold.
    fn devectorize_reduce_one(&mut self, reduce_id: OpId) {
        let Op::Reduce { x, rop, reduce_axis } = self.ops[reduce_id].op.clone() else {
            unreachable!("devectorize: not a reduce")
        };

        // The reduce spaces: every unroll node mentioned by the axis
        // expression, axis-ascending. The axis cone may only compose
        // unroll nodes with binary arithmetic and constants (e.g.
        // `unroll(axis0) * 4 + unroll(axis1)`); anything else is not a
        // lane space this pass reduces.
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
                    _ => todo!("devectorize: lane reduce over a non-unroll axis"),
                }
            }
        }
        if ordered.is_empty() {
            todo!("devectorize: reduce over no lane space");
        }
        ordered.sort_by_key(|&(_, _, axis)| axis);
        let space: Vec<(OpId, u16)> = ordered.into_iter().map(|(id, len, _)| (id, len)).collect();
        let reduce_len: u64 = space.iter().map(|&(_, l)| u64::from(l)).product();

        let (remaps, _) = self.duplicate_lanes(&space, DupMode::Reduce { reduce_id });

        // The reduce chain over per-lane values: the original (lane-0)
        // partial reduces with every later lane's clone of the value.
        let mut acc = x;
        for flat in 1..reduce_len {
            let lane_val = remaps.get(&x).map_or(x, |ids| ids[flat as usize - 1]);
            acc = self.insert_before(reduce_id, Op::Binary { x: acc, y: lane_val, bop: rop });
        }

        // Downstream consumers of the reduce now read the chain tail.
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

    /// Duplicate the ops that name one lane space, one clone per lane.
    ///
    /// `space` is `(node, len)` per axis, least-significant last: the unroll
    /// round passes one node, a lane reduce passes its axis expression's
    /// nodes. Lane 0 rewrites the originals in place, lanes `1..len` insert as
    /// clones after their producer. Returns the clones per op (the reduce's
    /// fold reads them) and the lane extracts repointed at a lane's copy.
    fn duplicate_lanes(&mut self, space: &[(OpId, u16)], mode: DupMode<'_>) -> (Map<OpId, Vec<OpId>>, Map<OpId, OpId>) {
        debug_assert!(!space.is_empty(), "devectorize: empty lane space");
        let total: u64 = space.iter().map(|&(_, len)| u64::from(len)).product();
        let mut strides: Vec<u64> = Vec::with_capacity(space.len());
        for i in 0..space.len() {
            let mut st = 1u64;
            for &(_, len) in &space[i + 1..] {
                st *= u64::from(len);
            }
            strides.push(st);
        }
        // One constant per lane per axis, inserted before its node so the
        // substitution can name it.
        let mut lane_consts: Vec<Vec<OpId>> = Vec::with_capacity(space.len());
        for &(node, len) in space {
            let mut consts = Vec::with_capacity(len as usize);
            for lane in 0..len {
                consts.push(self.insert_before(node, Op::Const(Constant::idx(lane))));
            }
            lane_consts.push(consts);
        }
        let lane_of = |flat: u64, k: usize| -> OpId {
            let len = u64::from(space[k].1);
            lane_consts[k][((flat / strides[k]) % len) as usize]
        };
        // The single node an unroll round splits, if this is one; and the
        // reduce being expanded, which never reduces through itself.
        let unroll = match mode {
            DupMode::Unroll { dup } => Some(dup),
            DupMode::Reduce { .. } => None,
        };
        let reduce_id = match mode {
            DupMode::Reduce { reduce_id } => reduce_id,
            DupMode::Unroll { .. } => OpId::NULL,
        };
        if unroll.is_some() {
            debug_assert_eq!(space.len(), 1, "devectorize: an unroll round splits one node");
        }

        // The per-op clones for lanes `1..total` (lane 0 rewrites the original).
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();
        // Lane extracts replaced by the lane's own copy.
        let mut aliases: Map<OpId, OpId> = Map::default();

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

        // The walk is linear in IR order so a producer's clones are registered
        // in `remaps` before its consumers are checked; the clones insert
        // after the current op, so the pointer `next` (its original successor)
        // never re-visits them.
        while !walk_id.is_null() {
            let next = self.next_op(walk_id);
            if walk_id == reduce_id {
                // Another reduce in the cone expands on its own round.
                walk_id = next;
                continue;
            }
            if unroll.is_some_and(|dup| !dup.contains(&walk_id)) {
                // Kept: the lane is a vector width here (or nothing of this
                // lane survives), so the op reads the lane unchanged. A
                // repointed extract is not the lane, so it is substituted
                // wherever it is read.
                if !aliases.is_empty() {
                    for param in self.ops[walk_id].op.parameters_mut() {
                        if let Some(&target) = aliases.get(param) {
                            *param = target;
                        }
                    }
                }
                walk_id = next;
                continue;
            }
            let op = self.ops[walk_id].op.clone();
            match op {
                Op::Storage { .. }
                | Op::Range { .. }
                | Op::Unroll { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::Param { .. }
                | Op::Const(_) => {}
                Op::Reduce { .. } => {
                    if unroll.is_some() {
                        unreachable!("reduce not devectorized before the unroll pass");
                    }
                    // Another reduce: independent of this lane space (neither
                    // naming its nodes nor consuming its clones) expands on
                    // its own round. A lane-conditional one shares the space
                    // in a way split cannot produce; that needs joint expansion
                    // and stays loud.
                    let lane_cond = self.ops[walk_id]
                        .op
                        .parameters()
                        .any(|p| space.iter().any(|&(node, _)| node == p) || remaps.contains_key(&p));
                    if lane_cond {
                        todo!("devectorize: reduce cone containing another reduce {walk_id}");
                    }
                }
                ref op => {
                    if unroll.is_some() {
                        // A lane extract of a value the lane is unfolding,
                        // when the source's lane is the innermost unit-stride
                        // term of its address: one element per lane. An
                        // instruction result or a `Stack` has component dims,
                        // never a lane, and a strided lane has pieces, not
                        // lanes. The extract names one fixed lane, so every
                        // read takes that lane's copy and the extract is
                        // dropped.
                        if let Op::Index { vec, idx } = *op
                            && let Some(clones) = remaps.get(&vec)
                            && self.lane_extract(vec, space[0].0)
                        {
                            let lane_op = if idx == 0 { Some(vec) } else { clones.get(idx - 1).copied() };
                            if let Some(lane_op) = lane_op {
                                remaps.insert(walk_id, vec![lane_op; total as usize - 1]);
                                aliases.insert(walk_id, lane_op);
                                walk_id = next;
                                continue;
                            }
                        }
                        // An mma fragment that the lane splits *into its
                        // slots* is one packed operand, not a work bundle:
                        // unfolding it hands each clone a single element where
                        // the instruction reads the packed slots (a `half2`
                        // operand becomes two `half` variables and the
                        // instruction reads the neighbour of the first). Pack
                        // the per-lane values back into the `Stack` the
                        // tensorizer emits and keep one instruction. A lane-
                        // shaped accumulator means the lane selects a whole
                        // instruction instead (one output tile per lane), so
                        // that case falls through and duplicates, each clone
                        // carrying its own fragment.
                        //
                        // The C operand is a fragment in every case: either
                        // its lanes *are* its slots (then they pack into one
                        // operand), or it already holds a whole fragment (a
                        // chained instruction's result, or the fragment after
                        // its tile axis was unrolled). Only a C that spans
                        // several instructions — one output tile per lane —
                        // means the lane selects instructions and the
                        // instruction duplicates.
                        if let Op::Wmma { dims, layout, dtype, c, a, b } = *op {
                            let conditional = |p: OpId| p == space[0].0 || remaps.contains_key(&p);
                            let len_dim: Dim = i64::try_from(total).expect("devectorize: lane count exceeds Dim");
                            let one_fragment = |k: &Self, p: OpId| {
                                k.wmma_operand_slots(walk_id, p).is_some_and(|slots| len_dim * k.elements(p) == slots)
                            };
                            let c_slots = self.wmma_operand_slots(walk_id, c);
                            let c_pack = conditional(c) && c_slots.is_some_and(|s| len_dim * self.elements(c) == s);
                            let c_whole = c_slots.is_some_and(|s| self.elements(c) == s);
                            let slots_packed = (!conditional(a) || one_fragment(self, a))
                                && (!conditional(b) || one_fragment(self, b))
                                && (!conditional(c) || c_pack || c_whole);
                            if slots_packed && (c_pack || conditional(a) || conditional(b) || (conditional(c) && c_whole)) {
                                let pack = |k: &mut Self, operand: OpId| {
                                    let mut ops = Vec::with_capacity(total as usize);
                                    ops.push(operand);
                                    if let Some(clones) = remaps.get(&operand) {
                                        ops.extend(clones.iter().copied());
                                    }
                                    debug_assert_eq!(ops.len() as u64, total, "devectorize: fragment lane count");
                                    k.insert_before(walk_id, Op::Stack { ops: ops.into() })
                                };
                                let a = if conditional(a) { pack(self, a) } else { a };
                                let b = if conditional(b) { pack(self, b) } else { b };
                                let c = if c_pack { pack(self, c) } else { c };
                                self.ops[walk_id].op = Op::Wmma { dims, layout, dtype, c, a, b };
                                walk_id = next;
                                continue;
                            }
                        }
                    }
                    let elem_cond = op.parameters().any(|p| space.iter().any(|&(node, _)| node == p) || remaps.contains_key(&p));
                    if !elem_cond {
                        walk_id = next;
                        continue;
                    }
                    let mut ids = Vec::with_capacity(total as usize - 1);
                    let mut at_id = walk_id;
                    for flat in 1..total {
                        let mut lane_op = op.clone();
                        // Reindex the op.
                        for param in lane_op.parameters_mut() {
                            if let Some(k) = space.iter().position(|&(node, _)| node == *param) {
                                *param = lane_of(flat, k);
                            } else if let Some(remap) = remaps.get(param) {
                                *param = remap[flat as usize - 1];
                            }
                        }
                        at_id = self.insert_after(at_id, lane_op);
                        ids.push(at_id);
                    }
                    // Lane 0 rewrites the original in place.
                    for param in self.ops[walk_id].op.parameters_mut() {
                        if let Some(k) = space.iter().position(|&(node, _)| node == *param) {
                            *param = lane_of(0, k);
                        } else if let Some(&target) = aliases.get(param) {
                            *param = target;
                        }
                    }
                    // Store which remaps will be used in the future.
                    remaps.insert(walk_id, ids);
                }
            }
            walk_id = next;
        }
        (remaps, aliases)
    }

    /// Unfold the dependent ops in `dup` per element: recreate each one once
    /// per element with the node substituted by that element's constant, and
    /// rewrite element 0 in place. The node stays when a kept op still reads
    /// it as a vector width; otherwise it is removed.
    /// A `Reduce` here is a driver bug: the joint loop expands reduces
    /// first, so none survive to the unroll rounds.
    fn devectorize_unroll_one(&mut self, unroll_id: OpId, dup: &Set<OpId>) {
        let len = match self.at(unroll_id) {
            Op::Unroll { len, .. } => *len,
            _ => unreachable!("devectorize: not an unroll node"),
        };
        let (_, aliases) = self.duplicate_lanes(&[(unroll_id, len)], DupMode::Unroll { dup });
        // The repointed extracts have no reader left.
        let extracts: Vec<OpId> = aliases.keys().copied().collect();
        for extract in extracts {
            self.remove_op(extract);
        }
        if self.iter_unordered().any(|(_, op)| op.parameters().any(|p| p == unroll_id)) {
            // Still a vector width for the kept dependents.
            return;
        }
        self.remove_op(unroll_id);
    }
}

impl Kernel {
    /// Slots the instruction expects in `operand` (`a`/`b`) of `wmma`.
    fn wmma_operand_slots(&self, wmma: OpId, operand: OpId) -> Option<Dim> {
        let Op::Wmma { dims, layout, dtype, a, b, c } = *self.at(wmma) else {
            return None;
        };
        let info = self.device_info();
        let v = info.mma_variants.iter().find(|v| v.dims == dims && v.layout == layout && v.dtype == dtype)?;
        // The operand's element dtype and the instruction's own slot count.
        // The device table owns both: the pass never knows what the elements
        // are, only that a word-valued operand packs `32 / bit size` of them.
        let (elem_dtype, operand_slots) = if operand == a {
            (v.a_dtype, v.a.slots.len() as Dim)
        } else if operand == b {
            (v.b_dtype, v.b.slots.len() as Dim)
        } else if operand == c {
            (v.c_dtype, v.c.slots.len() as Dim)
        } else {
            return None;
        };
        // `slots` counts the instruction's own elements. An `ldmatrix`
        // destination word carries several of them (two f16, four int8), so a
        // word-valued operand is `slots / elements-per-word` wide.
        if self.dtype(operand) == DType::U32 {
            let per_word = Dim::from(i64::from(32 / elem_dtype.bit_size()));
            Some(operand_slots / per_word)
        } else {
            Some(operand_slots)
        }
    }

    /// Element count of a value, from its shape.
    fn elements(&self, op: OpId) -> Dim {
        self.shape(op).iter().product()
    }

    /// The lane's coefficient in the *whole* index is one: the innermost
    /// contiguous element of the access.
    ///
    /// The coefficient, not the presence of the lane in some subterm, is what
    /// decides contiguity. `(2 * block + lane) * 76 + col` contains
    /// `Add(2 * block, lane)`, but the lane strides by 76 there: an eight-wide
    /// access would walk 76 elements per lane and read past the row.
    fn unit_lane_term(&self, idx: OpId, id: OpId) -> bool {
        // `(node, weight)`: the constant factor on the path from `idx` to
        // `node`. Sums add the weights, differences subtract, and a product by
        // a constant scales; anything else that carries the lane has no
        // constant coefficient and refuses the vector.
        let mut coeff: Dim = 0;
        let mut stack = vec![(idx, 1i64)];
        let mut seen = Set::default();
        while let Some((x, w)) = stack.pop() {
            if x == id {
                coeff += w;
                continue;
            }
            if !self.depends_on(x, id, &mut Set::default()) {
                continue;
            }
            // A lane-dependent node reached twice has no single coefficient
            // unless the paths are summed, and this walk does not track that.
            if !seen.insert(x) {
                return false;
            }
            match *self.at(x) {
                Op::Binary { x: a, y: b, bop: BOp::Add } => {
                    stack.push((a, w));
                    stack.push((b, w));
                }
                Op::Binary { x: a, y: b, bop: BOp::Sub } => {
                    stack.push((a, w));
                    stack.push((b, -w));
                }
                Op::Binary { x: a, y: b, bop: BOp::Mul } => {
                    let const_of = |v: OpId| self.resolve_const(v).and_then(Constant::as_dim);
                    match (const_of(a), const_of(b)) {
                        (Some(c), _) => stack.push((b, w * c)),
                        (None, Some(c)) => stack.push((a, w * c)),
                        // A runtime factor makes the stride unknown.
                        (None, None) => return false,
                    }
                }
                // A division, a mask or any other op that carries the lane:
                // the coefficient is not a constant.
                _ => return false,
            }
        }
        coeff == 1
    }

    /// `id` names a load, store or `ldmatrix` (with an address operand).
    fn is_access(&self, op: OpId) -> bool {
        match self.at(op) {
            Op::Load { .. } | Op::Store { .. } => true,
            // `ldmatrix` is the same access before and after
            // `lower_shared_addr`: the vector form names a storage, the
            // lowered form a byte address.
            Op::PTX(p) => matches!(p.as_ref(), PTXOp::VLdMatrix { .. } | PTXOp::LdMatrix { .. }),
            _ => false,
        }
    }

    /// The ops to unfold when lane `id` leaves the IR as scalars.
    ///
    /// Three clauses. (1) A value renders the lane when it is an access whose
    /// address names the lane as its one innermost unit-stride run the device
    /// supports (or whose address does not carry this lane at all, so it only
    /// carries the value through), with the store's value covering the width;
    /// everything else with no address renders. (2) A value is kept when it
    /// renders and its lane reaches an access through its consumers: a chain
    /// that ends in no memory traffic has no form to render. (3) Repetition is
    /// the connected component of everything that fails (2) — a kept value
    /// reads and is read by kept values only, because an unfolded neighbour
    /// hands it one element instead of the vector.
    ///
    /// Empty when the lane is a vector width throughout, i.e. when nothing
    /// needs unfolding.
    fn lane_expansion(&self, id: OpId) -> Set<OpId> {
        let Op::Unroll { len, .. } = *self.at(id) else {
            return Set::default();
        };
        let width = Dim::from(len);
        let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
        for (op, x) in self.iter_unordered() {
            for p in x.parameters() {
                if !p.is_null() {
                    consumers.entry(p).or_default().push(op);
                }
            }
        }
        // Every op transitively consuming the lane, in dependency order of
        // discovery (the loop below only needs the set).
        let mut dep: Set<OpId> = Set::default();
        let mut stack = vec![id];
        while let Some(x) = stack.pop() {
            for &c in consumers.get(&x).map_or(&[][..], |v| v.as_slice()) {
                if dep.insert(c) {
                    stack.push(c);
                }
            }
        }
        // A width the device cannot access as one vector is a thread/lane
        // bundle throughout: every dependent is scalar.
        let supported = self.device_info().supported_vec_lens.iter().any(|&v| Dim::from(v) == width);
        if !supported {
            return dep;
        }
        // Clause 2: a lane is kept only where it reaches an access. A chain
        // that ends in no memory traffic has nowhere to render the lane.
        let mut reaches: Set<OpId> = Set::default();
        loop {
            let mut changed = false;
            for &op in dep.iter() {
                if reaches.contains(&op) || !self.renders(op, id, width) {
                    continue;
                }
                let downstream =
                    self.is_access(op) || consumers.get(&op).is_some_and(|cs| cs.iter().any(|c| reaches.contains(c)));
                if downstream {
                    reaches.insert(op);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Repetition is the connected component of everything that does not
        // reach an access. A kept value reads and is read by kept values
        // only, because an unfolded neighbour hands it one element instead of
        // the vector.
        let mut repeat: Set<OpId> = Set::default();
        let mut stack: Vec<OpId> = Vec::new();
        for &op in &dep {
            if !reaches.contains(&op) {
                repeat.insert(op);
                stack.push(op);
            }
        }
        while let Some(x) = stack.pop() {
            for &c in consumers.get(&x).map_or(&[][..], |v| v.as_slice()) {
                if dep.contains(&c) && repeat.insert(c) {
                    stack.push(c);
                }
            }
            for p in self.at(x).parameters() {
                if !p.is_null() && dep.contains(&p) && repeat.insert(p) {
                    stack.push(p);
                }
            }
        }
        repeat
    }

    /// Whether the device renders `width` as one vector access or vector op.
    fn supported(&self, width: Dim) -> bool {
        self.device_info().supported_vec_lens.iter().any(|&v| Dim::from(v) == width)
    }

    /// The width of the vector access `idx` names for lane `id`: `id` is the
    /// innermost unit-stride term, no other lane competes for that role, and
    /// the device renders the width.
    fn vector_access(&self, idx: OpId, id: OpId) -> Option<Dim> {
        if idx.is_null() {
            return None;
        }
        let Op::Unroll { len, .. } = *self.at(id) else {
            return None;
        };
        let width = Dim::from(len);
        (self.supported(width) && self.vector_lane(idx, id)).then_some(width)
    }

    /// Whether `op` renders this lane as one vector width by itself.
    ///
    /// An access whose address does not depend on the lane is not addressing
    /// this lane: it carries the lane's value through, and renders.
    fn renders(&self, op_id: OpId, id: OpId, width: Dim) -> bool {
        let carries = |addr: OpId| self.depends_on(addr, id, &mut Set::default());
        match self.at(op_id) {
            Op::Load { src_idx, .. } => !carries(*src_idx) || self.vector_access(*src_idx, id) == Some(width),
            Op::Store { dst_idx, src, .. } => {
                !carries(*dst_idx)
                    || (self.vector_access(*dst_idx, id) == Some(width)
                        && (self.value_elements(*src) == width || self.value_elements(*src) == self.elements(*dst_idx)))
            }
            Op::PTX(p) => match p.as_ref() {
                PTXOp::VLdMatrix { addr, .. } | PTXOp::LdMatrix { addr, .. } => {
                    !carries(*addr) || self.vector_access(*addr, id) == Some(width)
                }
                _ => false,
            },
            // One scalar already; the unfolding gives it a clone per element.
            Op::Index { .. } => true,
            // A value with no address is a register value: codegen renders it
            // as a scalar, a vector or a tile.
            _ => true,
        }
    }

    /// Element count of a store's value: an `ldmatrix` result is the
    /// instruction's destination list (`num` 32-bit words) and has no lane
    /// shape of its own. The lowered form is the same fragment. Every other
    /// value carries its own shape.
    fn value_elements(&self, op: OpId) -> Dim {
        match self.at(op) {
            Op::PTX(p) => match p.as_ref() {
                PTXOp::VLdMatrix { num, .. } | PTXOp::LdMatrix { num, .. } => Dim::from(i64::from(num.words())),
                _ => self.elements(op),
            },
            _ => self.elements(op),
        }
    }

    /// The lane is the innermost unit-stride term of `idx` and no other lane
    /// competes for that role: one access has one innermost dimension, so a
    /// second unit-stride lane would widen the vector past the access.
    fn vector_lane(&self, idx: OpId, id: OpId) -> bool {
        if !self.unit_lane_term(idx, id) {
            return false;
        }
        let mut stack = vec![idx];
        let mut seen = Set::default();
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            if matches!(self.at(x), Op::Unroll { .. }) {
                if x != id && self.unit_lane_term(idx, x) {
                    return false;
                }
                continue;
            }
            for p in self.at(x).parameters() {
                if !p.is_null() {
                    stack.push(p);
                }
            }
        }
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
            mma_variants: vec![],
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

        let split = split::OptSplit { configs: vec![(loop_id, split::SplitMode::Unroll, 4)] };
        Optimization::apply(&split, &mut k, 0);
        k.devectorize().expect("small kernel devectorizes");

        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. })));
        k.verify();
    }

    fn unroll_kernel(len: u16) -> (Kernel, OpId, OpId, OpId, OpId) {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let lane = k.unroll(0, len);
        let stride = k.const_idx(4i64);
        (k, inp, out, lane, stride)
    }

    fn loads(k: &Kernel) -> usize {
        k.iter_unordered().filter(|(_, op)| matches!(op, Op::Load { .. })).count()
    }

    /// The lane is a work bundle, not a vector width, when the access walks it
    /// by its own step: every dependent unfolds to one scalar per lane.
    #[test]
    fn devectorize_repeats_a_strided_lane() {
        let (mut k, inp, out, lane, stride) = unroll_kernel(4);
        let off = k.mul(lane, stride);
        let v = k.load(inp, off);
        k.store(out, v, off);

        k.devectorize().expect("strided lane devectorizes");
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        assert_eq!(loads(&k), 4, "one scalar load per lane");
        k.verify();
    }

    /// The lane is a vector width when the access names it as the innermost
    /// unit-stride term and the device renders that width: it stays.
    #[test]
    fn devectorize_keeps_a_unit_stride_lane() {
        let (mut k, inp, out, lane, _) = unroll_kernel(4);
        let v = k.load(inp, lane);
        k.store(out, v, lane);

        k.devectorize().expect("unit-stride lane devectorizes");
        assert!(k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })), "the lane stays a vector width");
        assert_eq!(loads(&k), 1, "the access stays one vector load");
        k.verify();
    }

    /// Repetition is the connected component: a value that must repeat drags
    /// its producers and consumers with it, even those that would render
    /// alone. Here `unit` has a unit-stride address and the store does too,
    /// but both feed/consume the lane-strided `strided`, so all of it repeats.
    #[test]
    fn devectorize_repeats_the_connected_component() {
        let (mut k, inp, out, lane, stride) = unroll_kernel(4);
        let unit = k.load(inp, lane);
        let off = k.mul(lane, stride);
        let strided = k.load(inp, off);
        let sum = k.add(unit, strided);
        k.store(out, sum, lane);

        k.devectorize().expect("mixed lane devectorizes");
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        // 4 lanes x 2 loads, minus the lane-0 pair: both name address 0, and
        // the epilogue's CSE merges them.
        assert_eq!(loads(&k), 7, "both accesses unfold, one merged");
        let stores = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Store { .. })).count();
        assert_eq!(stores, 4, "the store is part of the component");
        k.verify();
    }

    /// A lane whose consumers reach no access has no form to render: it
    /// repeats too.
    #[test]
    fn devectorize_repeats_a_lane_with_no_access() {
        let (mut k, inp, out, lane, _) = unroll_kernel(4);
        let zero = k.const_val(0.0f32);
        k.store(out, zero, 0i64);
        let v = k.load(inp, lane);
        let one = k.const_val(1.0f32);
        let _dead = k.add(v, one);

        k.devectorize().expect("access-free lane devectorizes");
        assert!(!k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })));
        assert_eq!(loads(&k), 0, "the whole dead chain is gone");
        k.verify();
    }
}
