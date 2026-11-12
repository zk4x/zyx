// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Devectorize: unfold lane bundles left by `split` back to scalar IR.
//!
//! One pass, running in the device epilogue right before codegen, with
//! two mechanisms under a joint fixpoint: [`Kernel::devectorize`] expands
//! each lane-space reduce ([`Op::Reduce`] whose `reduce_axis` mentions
//! unroll nodes) into the reduce chain over its lanes, and unfolds every
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

use crate::{
    Map, Set, ZyxError,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, PTXOp},
    shape::Dim,
};

impl Kernel {
    /// Expand every lane-space reduce into the reduce chain over its lanes,
    /// then unfold every remaining unroll node into scalar statements.
    /// Unused nodes die in DCE. Runs in the device epilogue, before
    /// codegen. Reduces take priority every round: a reduce expansion can
    /// leave lane-conditional ops behind (axes the reduce does not
    /// consume), and the unroll mechanism unfolds those next rounds.
    ///
    /// The expansion creates fresh copies for each reduce lane; nothing
    /// runs after this joint with per-phase cleanups, so the reduced
    /// duplicates are cleaned once at the end: constant folding turns
    /// per-lane index expressions into literals, CSE merges the
    /// identical forms, DCE removes what is dead.
    pub fn devectorize(&mut self) -> Result<(), ZyxError> {
        if std::env::var_os("ZYX_DV").is_some() {
            self.debug();
        }
        let mut kept: Set<OpId> = Set::default();
        for _ in 0..10000 {
            // The expansion multiplies the IR by the lane count. Stop at the
            // render ceiling instead of handing the next pass a state it
            // cannot finish: constant folding is superlinear in op count, so
            // an oversized kernel turns the epilogue into minutes of host work
            // rather than a failure.
            let n_ops = self.ops.len().0;
            if n_ops > crate::kernel::MAX_OPS {
                return Err(ZyxError::kernel_error(
                    format!("devectorize: kernel IR has {n_ops} ops, above the {} -op limit", crate::kernel::MAX_OPS).into(),
                ));
            }
            // Upstream first (IR order, not unordered): expanding an inner
            // reduce before its consumers keeps nested reduces out of each
            // other's cones. Disjoint reduces don't care about order; a
            // genuinely shared lane space still halts loud in the walk.
            let mut found = OpId::NULL;
            let mut walk_id = self.head;
            while !walk_id.is_null() {
                if matches!(self.at(walk_id), Op::Reduce { .. }) {
                    found = walk_id;
                    break;
                }
                walk_id = self.next_op(walk_id);
            }
            if !found.is_null() {
                self.devectorize_reduce_one(found);
                continue;
            }
            let found =
                self.iter_unordered().find_map(|(op, x)| (matches!(x, Op::Unroll { .. }) && !kept.contains(&op)).then_some(op));
            let Some(unroll_id) = found else { break };
            // A lane is a vector width wherever the access it shapes is
            // contiguous and the device supports the width; only the
            // dependents that cannot carry it that way are unfolded. So one
            // lane may stay in the IR for its vector accesses while the same
            // lane's scalar consumers are expanded.
            let dup = self.lane_expansion(unroll_id);
            if dup.is_empty() {
                kept.insert(unroll_id);
                continue;
            }
            if !self.devectorize_unroll_one(unroll_id, &dup) {
                // The lane is still read as a vector width somewhere: it stays
                // in the IR, its scalar consumers are expanded.
                kept.insert(unroll_id);
            }
        }
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        if std::env::var_os("ZYX_DV_AFTER").is_some() {
            self.debug();
        }
        self.verify();
        Ok(())
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
        let reduce_nodes: Vec<(OpId, u16)> = ordered.into_iter().map(|(id, len, _)| (id, len)).collect();
        let mut strides: Vec<u64> = Vec::with_capacity(reduce_nodes.len());
        let reduce_len: u64 = reduce_nodes.iter().map(|&(_, l)| u64::from(l)).product();
        for i in 0..reduce_nodes.len() {
            let mut st = 1u64;
            for &(_, l) in &reduce_nodes[i + 1..] {
                st *= u64::from(l);
            }
            strides.push(st);
        }

        // Per-node lane constants; flat index `i` maps to node k's lane
        // `(i / stride_k) % len_k`, lane 0 first (the originals are
        // rewritten in place with lane 0).
        let mut lane_consts: Vec<Vec<OpId>> = Vec::with_capacity(reduce_nodes.len());
        for &(node_id, len) in &reduce_nodes {
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
        // 1..reduce_len (lane 0 rewrites the originals in place).
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();

        // Duplicate every reduce-conditional op once per remaining reduce
        // lane: ops directly naming the reduce's unroll nodes, and ops
        // consuming earlier-rewritten (remapped) ones. The reduce's own
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
                // The reduce being expanded never reduces through itself;
                // other reduces in its cone would need their own expansion
                // first.
                walk_id = next;
                continue;
            }
            match self.ops[walk_id].op.clone() {
                Op::Storage { .. }
                | Op::Range { .. }
                | Op::Unroll { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::Param { .. }
                | Op::Const(_) => {}
                Op::Reduce { .. } => {
                    // Another reduce: independent of this lane space (neither
                    // naming its nodes nor consuming its clones) expands on
                    // its own round. A lane-conditional one shares the space
                    // in a way split cannot produce; that needs joint expansion
                    // and stays loud.
                    let mut lane_cond = false;
                    for param in self.ops[walk_id].op.parameters() {
                        if reduce_nodes.iter().any(|&(f, _)| f == param) || remaps.contains_key(&param) {
                            lane_cond = true;
                            break;
                        }
                    }
                    if lane_cond {
                        todo!("devectorize: reduce cone containing another reduce {walk_id}");
                    }
                }
                ref op => {
                    let mut lane_cond = false;
                    for param in op.parameters() {
                        if reduce_nodes.iter().any(|&(f, _)| f == param) || remaps.contains_key(&param) {
                            lane_cond = true;
                            break;
                        }
                    }
                    if !lane_cond {
                        walk_id = next;
                        continue;
                    }
                    let mut ids = Vec::with_capacity(reduce_len as usize - 1);
                    let mut at_id = walk_id;
                    for flat in 1..reduce_len {
                        let mut lane_op = op.clone();
                        // Reindex the op.
                        for param in lane_op.parameters_mut() {
                            if let Some(ix) = reduce_nodes.iter().position(|&(f, _)| f == *param) {
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
                        if let Some(ix) = reduce_nodes.iter().position(|&(f, _)| f == *param) {
                            *param = lane_of(0, ix);
                        }
                    }
                    // Store which remaps will be used in the future.
                    remaps.insert(walk_id, ids);
                }
            }
            walk_id = next;
        }

        // The reduce chain over per-lane values: the original (lane-0)
        // partial reduces with every later lane's clone of the value.
        let mut acc = x;
        for flat in 1..reduce_len {
            let lane_val = match remaps.get(&x) {
                Some(ids) => ids[flat as usize - 1],
                None => x,
            };
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

    /// Unfold the dependent ops in `dup` per element: recreate each one once
    /// per element with the node substituted by that element's constant,
    /// rewrite element 0 in place, then drop the node when nothing reads it
    /// any more. Returns false when the node is still a vector width for the
    /// dependents outside `dup`.
    /// A `Reduce` here is a driver bug: the joint loop expands reduces
    /// first, so none survive to the unroll rounds.
    fn devectorize_unroll_one(&mut self, unroll_id: OpId, dup: &Set<OpId>) -> bool {
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
            if !dup.contains(&walk_id) {
                // Kept: the lane is a vector width here (or nothing of this
                // lane survives), so the op reads the lane unchanged.
                walk_id = next;
                continue;
            }
            match self.ops[walk_id].op.clone() {
                Op::Storage { .. }
                | Op::Range { .. }
                | Op::Unroll { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::Param { .. }
                | Op::Const(_) => {}
                Op::Reduce { .. } => unreachable!("reduce not devectorized before the unroll pass"),
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
                    // An mma fragment that the lane splits *into its slots* is
                    // one packed operand, not a work bundle: unfolding it hands
                    // each clone a single element where the instruction reads
                    // the packed slots (a `half2` operand becomes two `half`
                    // variables and the instruction reads the neighbour of the
                    // first). Pack the per-lane values back into the `Stack`
                    // the tensorizer emits and keep one instruction. A lane-
                    // shaped accumulator means the lane selects a whole
                    // instruction instead (one output tile per lane), so that
                    // case falls through and duplicates, each clone carrying
                    // its own fragment.
                    if let Op::Wmma { dims, layout, dtype, c, a, b } = *op {
                        let conditional = |p: OpId| p == unroll_id || remaps.contains_key(&p);
                        let len_dim: Dim = i64::try_from(len).expect("devectorize: unroll length exceeds Dim");
                        let one_fragment = |k: &Self, p: OpId| {
                            k.wmma_operand_slots(walk_id, p).is_some_and(|slots| len_dim * k.elements(p) == slots)
                        };
                        let slots_packed =
                            (!conditional(a) || one_fragment(self, a)) && (!conditional(b) || one_fragment(self, b));
                        if !conditional(c) && slots_packed && (conditional(a) || conditional(b)) {
                            let pack = |k: &mut Self, operand: OpId| {
                                let mut ops = Vec::with_capacity(len as usize);
                                ops.push(operand);
                                if let Some(clones) = remaps.get(&operand) {
                                    ops.extend(clones.iter().copied());
                                }
                                debug_assert_eq!(ops.len() as u64, len, "devectorize: fragment lane count");
                                k.insert_before(walk_id, Op::Stack { ops: ops.into() })
                            };
                            let a = if conditional(a) { pack(self, a) } else { a };
                            let b = if conditional(b) { pack(self, b) } else { b };
                            self.ops[walk_id].op = Op::Wmma { dims, layout, dtype, c, a, b };
                            walk_id = next;
                            continue;
                        }
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

        if self.iter_unordered().any(|(_, op)| op.parameters().any(|p| p == unroll_id)) {
            // Still a vector width for the kept dependents.
            return false;
        }
        self.remove_op(unroll_id);
        true
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
        if operand == a {
            Some(v.a.slots.len() as Dim)
        } else if operand == b {
            Some(v.b.slots.len() as Dim)
        } else if operand == c {
            Some(v.c.slots.len() as Dim)
        } else {
            None
        }
    }

    /// Element count of a value, from its shape.
    fn elements(&self, op: OpId) -> Dim {
        self.shape(op).iter().product()
    }

    /// The lane appears as an additive term with coefficient one in `idx`:
    /// the innermost contiguous element of the access.
    fn unit_lane_term(&self, idx: OpId, id: OpId) -> bool {
        if idx == id {
            // The lane is the whole index (base zero folded away).
            return true;
        }
        let mut stack = vec![idx];
        let mut seen = Set::default();
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            match *self.at(x) {
                Op::Binary { x: a, y: b, bop: BOp::Add } => {
                    if b == id && !self.depends_on(a, id, &mut Set::default()) {
                        return true;
                    }
                    if a == id && !self.depends_on(b, id, &mut Set::default()) {
                        return true;
                    }
                    stack.push(a);
                    stack.push(b);
                }
                Op::Binary { x: a, y: b, bop: BOp::Mul } => {
                    let one = |k: &Kernel, v: OpId| k.resolve_const(v).and_then(Constant::as_dim) == Some(1);
                    if (a == id && one(self, b)) || (b == id && one(self, a)) {
                        return true;
                    }
                    stack.push(a);
                    stack.push(b);
                }
                _ => {}
            }
        }
        false
    }

    /// `id` names a load, store or `ldmatrix` (with an address operand).
    fn is_access(&self, op: OpId) -> bool {
        match self.at(op) {
            Op::Load { .. } | Op::Store { .. } => true,
            Op::PTX(p) => matches!(p.as_ref(), PTXOp::VLdMatrix { .. }),
            _ => false,
        }
    }

    /// The ops to unfold when lane `id` leaves the IR as scalars.
    ///
    /// The lane is a *vector width* where the access it shapes is contiguous
    /// and the device supports the width, and a *work bundle* everywhere
    /// else. An op survives only when its own access carries the lane as a
    /// vector width and every op consuming it survives too; the rest of the
    /// lane's dependents are unfolded, together with the index math that
    /// feeds them (it would otherwise hand them a vector).
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
        // An op can stay only if its value still reaches an access: a chain
        // that ends in no memory traffic has nowhere to render the lane.
        let mut reaches: Set<OpId> = Set::default();
        loop {
            let mut changed = false;
            for &op in dep.iter() {
                if reaches.contains(&op) {
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
        let mut keep: Map<OpId, bool> = Map::default();
        for &op in dep.iter() {
            keep.insert(op, reaches.contains(&op) && self.lane_access_ok(op, id, width));
        }
        // A kept op must have only kept consumers: an unfolded consumer reads
        // one lane of a vector.
        loop {
            let mut changed = false;
            for &op in dep.iter() {
                if !keep[&op] {
                    continue;
                }
                let survives = consumers
                    .get(&op)
                    .is_none_or(|cs| cs.iter().all(|c| *keep.get(c).expect("devectorize: dependent op outside the lane's cone")));
                if !survives {
                    keep.insert(op, false);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // And a kept op must read only kept operands: an unfolded operand
        // hands it element 0, not the vector. Instructions are exempt — the
        // unroll pass repacks their per-lane operands into a `Stack`.
        loop {
            let mut changed = false;
            for &op in dep.iter() {
                if !keep[&op] || matches!(self.at(op), Op::Wmma { .. }) {
                    continue;
                }
                let operands_kept = self.at(op).parameters().all(|p| p == id || !dep.contains(&p) || keep[&p]);
                if !operands_kept {
                    keep.insert(op, false);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        dep.iter().filter(|op| !keep[*op]).copied().collect()
    }

    /// Whether `op` can render as one access of width `width` in lane `id`:
    /// the lane is the innermost contiguous term of the address, and the
    /// value side carries exactly that width (a broadcast scalar or a wider
    /// value is a work bundle, not a vector). An operand the lane does not
    /// shape at all is not this access' business.
    fn lane_access_ok(&self, op: OpId, id: OpId, width: Dim) -> bool {
        match self.at(op) {
            Op::Load { src_idx, .. } => {
                if !self.depends_on(*src_idx, id, &mut Set::default()) {
                    return true;
                }
                self.vector_lane(*src_idx, id) && self.lane_load_consumers_ok(op, width)
            }
            Op::Store { dst_idx, src, .. } => {
                if !self.depends_on(*dst_idx, id, &mut Set::default()) {
                    return true;
                }
                // The value carries this lane's width, or the whole access'
                // elements. The first is the single-lane case; the second is a
                // multi-lane access whose *other* lanes name addresses (a fill's
                // step lane over a vector width), where the value is one vector
                // covering the access and the backend splits it per address.
                // Comparing only against `width` refused every multi-lane
                // access, which is how a two-lane fill ended up scalarised.
                self.vector_lane(*dst_idx, id)
                    && (self.value_elements(*src) == width || self.value_elements(*src) == self.elements(*dst_idx))
            }
            Op::PTX(p) => match p.as_ref() {
                PTXOp::VLdMatrix { addr, .. } => !self.depends_on(*addr, id, &mut Set::default()) || self.vector_lane(*addr, id),
                _ => false,
            },
            _ => true,
        }
    }

    /// Element count of a store's value: an `ldmatrix` result is its
    /// fragment halves (`frags` 32-bit words, two halves each) and has no
    /// lane shape of its own.
    fn value_elements(&self, op: OpId) -> Dim {
        match self.at(op) {
            Op::PTX(p) => match p.as_ref() {
                PTXOp::VLdMatrix { frags, .. } => Dim::from(i64::from(*frags) * 2),
                _ => Dim::from(2),
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

    /// Every consumer of a vector load must take the whole width: a
    /// per-element extract (`Index`), a copy (`Store`) or an instruction
    /// whose slot count is exactly the width (`Wmma`, through the `Stack`
    /// that packs its operand).
    fn lane_load_consumers_ok(&self, load: OpId, width: Dim) -> bool {
        for (cid, cop) in self.iter_unordered() {
            if !cop.parameters().any(|p| p == load) {
                continue;
            }
            match *cop {
                Op::Index { vec, .. } if vec == load => {}
                Op::Store { src, .. } if src == load => {}
                Op::Wmma { .. } => {
                    if self.wmma_operand_slots(cid, load) != Some(width) {
                        return false;
                    }
                }
                Op::Stack { ref ops } => {
                    let total: Dim = ops.iter().map(|&o| self.elements(o)).sum();
                    let fragment = self.iter_unordered().any(|(wid, wop)| {
                        matches!(wop, Op::Wmma { a, b, .. }
                            if (*a == cid || *b == cid)
                                && self.wmma_operand_slots(wid, cid) == Some(total))
                    });
                    if !fragment {
                        return false;
                    }
                }
                _ => return false,
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
}
