// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Rangeify movement operations.
//!
//! Reimplements unfold_movement_ops
//! without the View abstraction for movement op propagation.
//! Movement ops are applied directly to axis indices, and
//! LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.
//!
//! # Kernel structure before and after linearize
//!
//! Before [`linearize`](Kernel::linearize), kernels contain only high-level ops:
//! `Param`, `Move` (reshape/expand/permute/pad/flip), `Reduce`, `Binary`, and
//! `Store`. Notably, they contain **no `Load`s**. All inputs to a kernel are
//! `Op::Param` with `ParamKind::Global`/`GlobalMut` kind, and every global param is either:
//!
//! - **read-only** (`ro: true`) — an input, later turned into a `Load`
//! - **not read-only** (`ro: false`) — a `Store` destination (an output)
//!
//! Because there are no `Load`s before linearize, the kernel does not yet have a
//! `loads` list.
//!
//! Linearize performs the bulk of the "unfolding":
//!
//! - it **removes `Move` and `Reduce`**, expanding them into index arithmetic,
//! - it **inserts `Load`s**, `Loop`s, and the indexing computation (`Index`,
//!   `Mad`, `Binary` on loop/group indices),
//! - it computes each `Store`'s index from the shape it writes,
//! - read-only global params become `Load { src, .. }` referencing a freshly
//!   inserted source `Storage`,
//! - writable global params stay in place as `Store` destinations.
//!
//! `Storage` is a post-linearization operation. It must not be expected in
//! pre-linearization movement kernels or used as a movement-chain marker.
//!
//! Only after linearize does the kernel have `Load` ops, so the `loads` list only
//! becomes meaningful then. This matters for any pass that maps kernel args to
//! buffers: pre-linearize, map from the global `Param` ops (in op order), not
//! from a `loads` list.

/// A single symbolic dimension of a value's index view: the loop/group index
/// (`idx`) and axis length (`len`). `len` is the literal shape of the op the
/// view belongs to. All are `OpId`s resolved lazily.
#[derive(Clone, Copy)]
pub(crate) struct SDim {
    pub(crate) idx: OpId,
    pub(crate) len: OpId,
}

impl SDim {
    pub(crate) fn new(idx: OpId, len: OpId) -> Self {
        Self { idx, len }
    }
}

use std::collections::BinaryHeap;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind},
    shape::{Dim, UAxis},
};

impl Kernel {
    /// Unfold movement operations into index-based operations
    ///
    /// Movement ops (Reshape, Expand, Permute, Pad) are applied directly to axis indices,
    /// and LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.
    // TODO Currently it only works if each param has a single move op chain.
    // Make it also work with move op chains when each param is accessed by multiple move ops.
    pub fn linearize(&mut self) {
        if !self.ops.values().any(|n| matches!(n.op, Op::Store { index: OpId::NULL, .. })) {
            return;
        }

        // Duplicating multi-use constants repoints `Param { shape }` fields at
        // fresh constants, invalidating the memoized `shape_ids`; drop the
        // cache so `add_indexing` re-derives shapes from the actual
        // (repointed) parameters.
        self.shape_cache = Map::default();
        self.duplicate_multi_use_consts();

        #[cfg(debug_assertions)]
        {
            let has_gidx = self.ops.values().any(|n| matches!(n.op, Op::Index { kind: IdxKind::Group(_), .. }));
            let has_moves = self.ops.values().any(|n| matches!(n.op, Op::Move { .. }));
            if has_gidx && has_moves {
                panic!("unfold_movement_ops: cannot have both explicit gidx and LoadView/StoreView/Move ops");
            }
        }

        debug_assert!({
            let mut live: Set<OpId> = Set::default();
            let mut stack: Vec<OpId> = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if matches!(self.ops[op_id].op, Op::Store { .. }) {
                    stack.push(op_id);
                }
                op_id = self.next_op(op_id);
            }
            while let Some(id) = stack.pop() {
                if !id.is_null() && live.insert(id) {
                    stack.extend(self.ops[id].op.parameters());
                }
            }
            op_id = self.head;
            while !op_id.is_null() {
                if !live.contains(&op_id) {
                    self.debug();
                    panic!("linearize: dead code detected at op {op_id}");
                }
                op_id = self.next_op(op_id);
            }
            true
        });

        // Snapshot the order of global params so linearize can assert it never
        // reorders the buffers' declaration order.
        let global_params: Vec<(DType, ParamKind)> = {
            let mut params = Vec::new();
            let mut op_id = self.head;
            for _ in 0..50_000 {
                if op_id.is_null() {
                    break;
                }
                if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                    params.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            if !op_id.is_null() {
                panic!("linearize did not finish in 50000 steps");
            }
            params
        };

        self.add_indexing();

        // After linearization the parameter shapes are no longer meaningful;
        // clear them so the verify below (and later passes) don't require shape
        // consts to be ordered before the params that reference them.
        for node in self.ops.values_mut() {
            if let Op::Param { shape, .. } = &mut node.op {
                *shape = OpId::NULL;
            }
        }

        // Read-only (Variable + Global) and writable (GlobalMut) params in
        // linked-list order. Since Phase 1 does not reorder params, forward order
        // is the correct kernel argument order.
        let mut ro_params: Vec<OpId> = Vec::new();
        let mut rw_params: Vec<OpId> = Vec::new();
        {
            let mut op_id = self.head;
            for _ in 0..50_000 {
                if op_id.is_null() {
                    break;
                }
                if let Op::Param { kind, .. } = self.ops[op_id].op {
                    match kind {
                        ParamKind::Variable | ParamKind::Global => ro_params.push(op_id),
                        ParamKind::GlobalMut => rw_params.push(op_id),
                    }
                }
                op_id = self.next_op(op_id);
            }
            if !op_id.is_null() {
                panic!("linearize did not finish in 50000 steps");
            }
        }
        self.toposort(&ro_params, &rw_params);

        // Verify the relative order of global params is unchanged by linearize
        // (read-only params first, then writable ones, both in original order).
        debug_assert!({
            let mut params = Vec::new();
            let mut op_id = self.head;
            for _ in 0..50_000 {
                if op_id.is_null() {
                    break;
                }
                if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                    params.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            let mut expected = global_params.clone();
            expected.sort_by_key(|(_, kind)| *kind == ParamKind::GlobalMut);
            if params != expected {
                self.debug();
                panic!(
                    "linearize: global param order changed:\n  original = {global_params:?}\n  expected = {expected:?}\n  final = {params:?}"
                );
            }
            true
        });

        self.autocast_scalars();

        self.add_control_flow();

        //
        // The move handlers may leave dead constants (e.g. unused `one`/`total`
        // scaffold) and duplicate arithmetic behind; CSE and DCE clean those up
        // now that the ops are ordered.
        assert!(
            self.ops.values().all(|node| !matches!(node.op, Op::Move { .. } | Op::Stack { .. })),
            "linearize left a movement or stack operation in the kernel"
        );

        // Dedup group-index ops: every store handler emits its own set of
        // `Op::Index { kind: Group }` per output axis, so N stores of the same
        // shape produce N duplicate group indices per axis. Keep the first in
        // linked-list order as canonical and remap the rest onto it. All
        // duplicates for an axis must agree on the length, or the kernel is
        // malformed.
        {
            let mut canonical: Map<u32, OpId> = Map::default();
            let mut lengths: Map<u32, Dim> = Map::default();
            let mut op_id = self.head;
            for _ in 0..50_000 {
                if op_id.is_null() {
                    break;
                }
                let next = self.next_op(op_id);
                if let Op::Index { axis, kind: IdxKind::Group(len) } = self.ops[op_id].op {
                    let len_dim = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap_or(i64::MAX as Dim);
                    if let Some(&l) = lengths.get(&axis) {
                        assert!(len_dim == l, "group index axis={axis} has inconsistent lengths ({} vs {})", l, len_dim);
                        self.remap(op_id, canonical[&axis]);
                        self.remove_op(op_id);
                    } else {
                        lengths.insert(axis, len_dim);
                        canonical.insert(axis, op_id);
                    }
                }
                op_id = next;
            }
            if !op_id.is_null() {
                panic!("linearize did not finish in 50000 steps");
            }
        }

        self.verify();
        self.common_subexpression_elimination();
        self.dead_code_elimination();

        // The shape_ids cache is only valid pre-linearization; drop it so
        // autotuned kernels stay free of cached shape scaffolding.
        self.shape_cache = Map::default();
    }

    /// Duplicates multi-use constants so every `Op::Const` ends up with
    /// exactly one use.
    ///
    /// The eager fusion path (`duplicate_or_store`) intentionally duplicates
    /// values consumed under different indexing/loop schemes: after
    /// linearization a value computed inside one loop scope cannot be
    /// referenced from another, because the scope's declaration set is popped
    /// at `EndLoop`. Kernel merging may however collapse identical constants
    /// from contributing kernels into one shared op, leaving a single `Const`
    /// whose users sit in different scopes — linearize would schedule it in
    /// one scope while a user lives in another, and verify rejects the
    /// resulting use-before-declaration.
    ///
    /// Constants are pure leaves, so duplicating them is always semantically
    /// exact: each use gets its own copy and linearize schedules every copy
    /// in its user's scope. Runs after the kernel cache lookup, so the extra
    /// ops never reach the cache key.
    fn duplicate_multi_use_consts(&mut self) {
        // Phase 1: count references per op over the linked list.
        let mut use_count: Map<OpId, u32> = Map::default();
        let mut op_id = self.head;
        for _ in 0..50_000 {
            if op_id.is_null() {
                break;
            }
            for param in self.ops[op_id].op.parameters() {
                *use_count.entry(param).or_default() += 1;
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("duplicate_multi_use_consts did not finish in 50000 steps");
        }

        // Only constants referenced more than once need duplication; the
        // original keeps its last use in chain order, every earlier use gets
        // a fresh copy. No dead ops are created.
        let mut multi: Map<OpId, (Constant, u32)> = Map::default();
        for (&id, &count) in use_count.iter() {
            if count > 1 {
                if let Op::Const(value) = self.ops[id].op {
                    multi.insert(id, (value, count - 1));
                }
            }
        }
        if multi.is_empty() {
            return;
        }

        // Phase 2: repoint each extra use at a fresh constant inserted
        // directly before its user. Constants are pure leaves, so this is
        // always in topological order.
        let mut op_id = self.head;
        for _ in 0..50_000 {
            if op_id.is_null() {
                break;
            }
            let next = self.next_op(op_id);
            let params: Vec<OpId> = self.ops[op_id].op.parameters().collect();
            for (position, param) in params.into_iter().enumerate() {
                let Some(entry) = multi.get_mut(&param) else {
                    continue;
                };
                if entry.1 == 0 {
                    // Last use in chain order keeps the original constant.
                    continue;
                }
                entry.1 -= 1;
                let value = entry.0;
                let fresh = self.insert_before(op_id, Op::Const(value));
                let Some(param) = self.ops[op_id].op.parameters_mut().nth(position) else {
                    panic!("duplicate_multi_use_consts: param position {position} out of range for op {op_id:?}");
                };
                *param = fresh;
            }
            op_id = next;
        }
        if !op_id.is_null() {
            panic!("duplicate_multi_use_consts did not finish in 50000 steps");
        }
    }

    /// Inserts index arithmetic (views, strides, pads, bounds checks) for
    /// every op. Runs after phase 1: from here on shapes are not parameters
    /// of ops anymore — at the end of `add_indexing` every `Param { shape }`
    /// is set to null, because indexing replaces explicit shape stacks and
    /// they are not needed anymore. Operand dtypes may also be inconsistent
    /// at this point; `autocast_scalars` resolves them later.
    fn add_indexing(&mut self) {
        // Shared zero/one index constants used throughout the handlers, hoisted
        // once so every branch reuses them instead of inserting fresh constants.
        let zero = self.const_idx(0);
        let one = self.const_idx(1);

        // For each op, shape and strides: (index, stride, left pad, right pad, axis length)
        let mut views: Map<OpId, Vec<SDim>> = Map::default();

        // Maps a writable global param to the store that writes into it. The
        // store handler records the entry (walking dst through any moves to the
        // terminal storage); the param handler uses it to write back the store's
        // computed index.
        let mut dst_stores: Map<OpId, OpId> = Map::default();

        // Snapshot the original ops in list order. Handlers insert index arithmetic
        // before `start` or the innermost open-loop anchor; walking a snapshot in
        // reverse avoids processing those inserted ops (they are not view ops and
        // have no view entry).
        // Collect ops reachable from the store roots. Ops not on any store's
        // dependency chain are dead and must be skipped during the reverse walk
        // (their views are never seeded, and touching them would panic).
        let mut roots: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        for _ in 0..50_000 {
            if op_id.is_null() {
                break;
            }
            if matches!(self.ops[op_id].op, Op::Store { .. }) {
                roots.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("add_indexing did not finish in 10000 steps");
        }
        let mut reachable = Set::default();
        let mut pending = roots;
        for _ in 0..50_000 {
            let Some(op_id) = pending.pop() else { break };
            if self.ops.contains_id(op_id) {
                if reachable.insert(op_id) {
                    pending.extend(self.at(op_id).parameters());
                }
            }
        }
        if !pending.is_empty() {
            panic!("add_indexing did not finish in 10000 steps");
        }
        let mut op_ids: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        for _ in 0..50_000 {
            if op_id.is_null() {
                break;
            }
            if reachable.contains(&op_id) {
                op_ids.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("add_indexing did not finish in 10000 steps");
        }
        // Phase 1: unfold movement ops (reshape/narrow/...) into index
        // arithmetic, converting LoadView/StoreView/ConstView into Load/Store/
        // Const. The inserted arithmetic is appended at anchor positions; its
        // relative order is fixed up in Phase 2. Reductions are left intact
        // here (their loops are emitted in Phase 3).
        for &op_id in op_ids.iter().rev() {
            // Leave loop scopes as the reverse walk exits them. Popped after the
            // op is processed: the loop_start op itself sits inside the loop (it
            // is the first dependency of the reduce input), so its own inserted
            // index arithmetic must still land inside the loop.
            match self.ops[op_id].op {
                Op::Const(value) => {
                    let Some(view) = views.remove(&op_id) else { continue };
                    // The constant is a scalar whose value must be nullified where the
                    // view's bounds condition is false (padded regions read as zero).
                    // len is the op's literal shape, so the plain bounds check
                    // idx >= 0 && idx < len is exact; pads that are Const(0) simply
                    // fold away in later passes.
                    let mut pc = self.const_val(true);
                    for d in &view {
                        let t_lo = self.cmpge(d.idx, zero);
                        pc = self.and(t_lo, pc);
                        // A dim length of 0 is the inferred-dim marker and must
                        // never reach the kernel IR (Tensor::reshape rejects it).
                        debug_assert!(
                            self.resolve_const(d.len).and_then(Constant::as_dim) != Some(0),
                            "inferred dim (0) must not reach linearize"
                        );
                        let t_hi = self.cmplt(d.idx, d.len);
                        pc = self.and(t_hi, pc);
                    }
                    let z = self.push_back(Op::Const(value));
                    self.ops[op_id].op = Op::Binary { x: pc, y: z, bop: BOp::Mul };
                }
                Op::Param { dtype, kind, shape } => {
                    // Metadata-only param: referenced only as a shape descriptor
                    // (never loaded as data), so no view was seeded for it.
                    // Load paths only — a shaped param must always be loaded.
                    if matches!(kind, ParamKind::Global | ParamKind::Variable) && !views.contains_key(&op_id) {
                        debug_assert!(shape.is_null(), "viewless param must be a scalar, got {shape:?}");
                        continue;
                    }
                    match kind {
                        // Register-scope storages (e.g. reduce accumulators) are managed
                        // by the ops that create them; only global params are
                        // rangeified here. Writable globals are store destinations,
                        // read-only globals/variables are load sources. Writables with
                        // MemScope::Variable are left alone (stores to variables are
                        // invalid; the verifier rejects them).
                        ParamKind::GlobalMut => {
                            // Write path: this param is the destination of a store. The
                            // store's index is computed from the param's rangeified view
                            // and written back into the matching store op.
                            let store_id = dst_stores.remove(&op_id).unwrap();
                            let view = views.remove(&op_id).unwrap();
                            // len is the literal shape of this op, so row-major contiguous
                            // strides are derived directly from it (no stored stride).
                            let mut write_index = zero;
                            let mut stride = one;
                            let mut strides = Vec::with_capacity(view.len());
                            for d in view.iter().rev() {
                                strides.push(stride);
                                stride = self.mul(stride, d.len);
                            }
                            strides.reverse();
                            for (d, s) in view.iter().zip(strides) {
                                write_index = self.mad(d.idx, s, write_index);
                            }
                            match &mut self.ops[store_id].op {
                                Op::Store { index, .. } => *index = write_index,
                                _ => unreachable!("graph stores are the only stores at linearize time"),
                            }
                        }
                        ParamKind::Variable => {
                            let view = views.remove(&op_id).unwrap();
                            // Variables are single values (no indexing). Like constants,
                            // they only need the padding mask: where the view is out of
                            // bounds, the loaded value is zeroed.
                            let mut pc = self.const_val(true);
                            for d in &view {
                                let t_lo = self.cmpge(d.idx, zero);
                                pc = self.and(t_lo, pc);
                                // A dim length of 0 is the inferred-dim marker and must
                                // never reach the kernel IR (Tensor::reshape rejects it).
                                debug_assert!(
                                    self.resolve_const(d.len).and_then(Constant::as_dim) != Some(0),
                                    "inferred dim (0) must not reach linearize"
                                );
                                let t_hi = self.cmplt(d.idx, d.len);
                                pc = self.and(t_hi, pc);
                            }
                            // A variable IS its value: like a constant it needs only
                            // the pad mask — no storage insert, no load. A fresh
                            // param is inserted so the define order (which scalar
                            // args bind to) is preserved, then the value is
                            // multiplied by the mask.
                            let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                            self.ops[op_id].op = Op::Binary { x: pc, y: src, bop: BOp::Mul };
                        }
                        ParamKind::Global => {
                            let view = views.remove(&op_id).unwrap();
                            // Bounds condition: valid where index is within the source
                            // extent. `len` is the literal shape, so the plain bounds
                            // check idx >= 0 && idx < len is exact; every movement op
                            // bakes its shift into `idx` and adjusts `len` (tinygrad's
                            // model), so no separate pad terms are needed.
                            //   index = sum over axes of idx * stride
                            //   pc    = and over axes of (idx >= 0) && (idx < len)
                            let mut index = self.const_idx(0);
                            let mut pc = self.const_val(true);
                            let mut stride = one;
                            let mut strides = Vec::with_capacity(view.len());
                            for d in view.iter().rev() {
                                strides.push(stride);
                                stride = self.mul(stride, d.len);
                            }
                            strides.reverse();
                            for (d, s) in view.iter().zip(strides) {
                                index = self.mad(d.idx, s, index);
                                let ge = self.cmpge(d.idx, zero);
                                pc = self.and(ge, pc);
                                // A dim length of 0 is the inferred-dim marker and must
                                // never reach the kernel IR (Tensor::reshape rejects it).
                                debug_assert!(
                                    self.resolve_const(d.len).and_then(Constant::as_dim) != Some(0),
                                    "inferred dim (0) must not reach linearize"
                                );
                                let lt = self.cmplt(d.idx, d.len);
                                pc = self.and(lt, pc);
                            }
                            // Insert the ro source storage immediately before this op so the
                            // global param order (which buffer args bind to) is preserved.
                            let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                            // Zero the offset where the padding condition fails, so the load
                            // always reads in-bounds, then zero the loaded value itself.
                            let offset = self.mul(pc, index);
                            let z = self.load(src, offset, MemLayout::Scalar);
                            self.ops[op_id].op = Op::Binary { x: pc, y: z, bop: BOp::Mul };
                        }
                    }
                }
                Op::Store { dst, src, index, layout } => {
                    debug_assert_eq!(index, OpId::NULL);
                    debug_assert_eq!(layout, MemLayout::Scalar);
                    // The store writes its dst op's whole view. Loop lengths come
                    // from the dst op's own shape — NOT the terminal Param's shape:
                    // a crop (`pad lp<0`) or narrow between the Param and the store
                    // makes the view smaller than the backing buffer, and looping
                    // over the Param shape would run past the view (reading OOB and
                    // clobbering elements adjacent to the view). The move handlers
                    // below (pad/narrow/...) then shift the index into the base
                    // domain; the Param handler computes the flat write index from
                    // that shifted view.
                    let mut dst_param = dst;
                    for _ in 0..50_000 {
                        let Op::Move { x, .. } = self.ops[dst_param].op else { break };
                        dst_param = x;
                    }
                    if matches!(self.ops[dst_param].op, Op::Move { .. }) {
                        panic!("add_indexing store dst chain did not finish in 10000 steps");
                    }
                    let dst_param_op = &self.ops[dst_param].op;
                    assert!(
                        matches!(dst_param_op, Op::Param { kind: ParamKind::GlobalMut, .. }),
                        "store dst chain must terminate at a writable global Param, got {dst_param_op:?}"
                    );
                    let Op::Param { .. } = *dst_param_op else { unreachable!() };
                    assert!(
                        dst_stores.insert(dst_param, op_id).is_none(),
                        "store dst chain terminates at Param {dst_param:?}, which is already a store destination"
                    );
                    self.ops[op_id].op = Op::Store { dst, src, index: OpId::NULL, layout: MemLayout::Scalar };
                    let dims = self.shape_ids(dst);
                    let mut view = Vec::new();
                    for (axis, &len) in dims.iter().enumerate().rev() {
                        let idx = self.group_index(axis as u32, len);
                        view.push(SDim::new(idx, len));
                    }
                    view.reverse();
                    views.insert(src, view.clone());
                    views.insert(dst, view);
                }
                Op::Reduce { x, rop, reduce_axis } => {
                    // Build the reduce input x's view with all dims: the non-reduced
                    // dims come from the reduce output's view (set by the Store
                    // handler), and the reduced (last) dim is a freshly-opened loop
                    // over `reduce_axis`. The reduce's `reduce_axis` is repointed at
                    // that loop so the loop gets ordered before the reduce's input
                    // computation in Phase 2 (outer loops before inner ones).
                    let out_view = views.remove(&op_id).unwrap();
                    let loop_id = self.loop_(reduce_axis);
                    // Non-reduced axes must use the reduce input `x`'s row-major
                    // strides (idx/lp/rp/len come from the output view, stride is
                    // recomputed from the input shape, which includes the reduced
                    // axis). The reduced axis is the freshly-opened loop with the
                    // input's contiguous stride and zero padding.
                    let x_shape = self.shape_ids(x);
                    let n = x_shape.len();
                    let non_reduce = out_view.len();
                    let mut view = Vec::with_capacity(n);
                    for d in out_view {
                        view.push(SDim::new(d.idx, d.len));
                    }
                    for a in non_reduce..n {
                        view.push(SDim::new(loop_id, x_shape[a]));
                    }
                    views.insert(x, view);
                    self.ops[op_id].op = Op::Reduce { x, rop, reduce_axis: loop_id };
                }
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { .. } => {
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. The input is read as a single flat index over the
                            // whole (contiguous) input, which equals the flat index over the
                            // output. Build `base` from the output view using each axis's
                            // stored stride, then recover each input axis by successive
                            // div/mod against the input's contiguous strides. Any pad/crop
                            // offsets on the reshape output's axes have already been baked
                            // into `d.idx` by the pad handlers, so no lp handling is needed
                            // here.
                            let out_view = views[&op_id].clone();
                            let x_shape = self.shape_ids(x);
                            let n = x_shape.len();
                            let mut x_strides = vec![one; n];
                            let mut st = one;
                            for a in (0..n).rev() {
                                x_strides[a] = st;
                                st = self.mul(x_shape[a], st);
                            }
                            // Validity mask over the output view: a recovered input
                            // coordinate is only meaningful where the output is within
                            // its own source extent (idx >= 0 && idx < len).
                            // Padded output regions must read as zero, so invalid
                            // recovered indices are clamped to len + 1 (out of bounds).
                            let mut valid = self.const_val(true);
                            for d in &out_view {
                                let lo = self.cmpge(d.idx, zero);
                                // A dim length of 0 is the inferred-dim marker and must
                                // never reach the kernel IR (Tensor::reshape rejects it).
                                debug_assert!(
                                    self.resolve_const(d.len).and_then(Constant::as_dim) != Some(0),
                                    "inferred dim (0) must not reach linearize"
                                );
                                let hi = self.cmplt(d.idx, d.len);
                                let in_axis = self.and(lo, hi);
                                valid = self.and(valid, in_axis);
                            }
                            let mut base = zero;
                            let mut stride = one;
                            let mut out_strides = Vec::with_capacity(out_view.len());
                            for d in out_view.iter().rev() {
                                out_strides.push(stride);
                                stride = self.mul(stride, d.len);
                            }
                            out_strides.reverse();
                            for (d, s) in out_view.iter().zip(out_strides) {
                                base = self.mad(d.idx, s, base);
                            }
                            let mut view = Vec::with_capacity(n);
                            let mut q = base;
                            for a in 0..n {
                                let s = x_strides[a];
                                let idx_expr = if a == n - 1 {
                                    q
                                } else {
                                    let div = self.div(q, s);
                                    let rem = self.mod_(q, s);
                                    q = rem;
                                    div
                                };
                                let len = x_shape[a];
                                let invalid = self.add(len, one);
                                let idx_expr = self.branchless_where(valid, idx_expr, invalid);
                                view.push(SDim::new(idx_expr, len));
                            }
                            views.insert(x, view);
                        }
                        &MoveOp::Expand { .. } => {
                            // Broadcast determination is symbolic: an input axis is
                            // broadcast iff its dim resolves to 1 and the output dim
                            // resolves to something != 1 (mirrors tinygrad's
                            // broadcast_axes/resolve). A dynamic dim resolves to None
                            // and is treated as non-broadcast (identity), the safe
                            // default. No concrete shape() lookup is required.
                            let x_shape = self.shape_ids(x);
                            let shape = match &self.ops[op_id].op {
                                Op::Move { mop, .. } => match mop.as_ref() {
                                    MoveOp::Reshape { shape, .. } | MoveOp::Expand { shape } => match &self.ops[*shape].op {
                                        Op::Stack { ops } => ops.to_vec(),
                                        // Bare descriptor: a single dim value (const,
                                        // runtime-loaded scalar, or a dim *expression*
                                        // over them) — mirrors `shape_ids`'s `descriptor`.
                                        Op::Const(_)
                                        | Op::Param { .. }
                                        | Op::Unary { .. }
                                        | Op::Binary { .. }
                                        | Op::Load { .. } => {
                                            vec![*shape]
                                        }
                                        op => todo!("invalid shape descriptor {op:?}"),
                                    },
                                    _ => unreachable!(),
                                },
                                _ => unreachable!(),
                            };
                            // New leading axes are prepended broadcasts; the input axes
                            // align to the tail of the output shape. A broadcast input
                            // axis reads a single constant element (index 0 over an
                            // input length of 1); a non-broadcast axis keeps the input's
                            // own index and length, so the load indexes the compact input.
                            let offset = shape.len() - x_shape.len();
                            let n = x_shape.len();
                            let out_view = views[&op_id].clone();
                            let view = if n == 0 {
                                // Scalar input broadcasts to every axis: the input view
                                // is the whole output view, so the pad mask propagates.
                                out_view
                            } else {
                                let mut v = Vec::with_capacity(n);
                                for a in 0..n {
                                    let broadcast = self.resolve_const(x_shape[a]).and_then(Constant::as_dim) == Some(1)
                                        && self.resolve_const(shape[offset + a]).and_then(Constant::as_dim) != Some(1);
                                    let d = out_view[offset + a];
                                    let d = if broadcast {
                                        SDim::new(zero, x_shape[a])
                                    } else {
                                        SDim::new(d.idx, x_shape[a])
                                    };
                                    v.push(d);
                                }
                                v
                            };
                            views.insert(x, view);
                        }
                        MoveOp::Permute { axes } => {
                            // Pure backwards permutation: input axis j is consumed by
                            // output axis inv_axes[j], so the input view's axis j is
                            // exactly the output view's axis inv_axes[j]. No shape
                            // lookup or stride recomputation is needed -- the SDims are
                            // simply reordered.
                            let axes = axes.clone();
                            let view = views[&op_id].clone();
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a] = i;
                            }
                            let view: Vec<SDim> = inv_axes.iter().map(|&j| view[j]).collect();
                            views.insert(x, view);
                        }
                        MoveOp::Flip { axes } => {
                            let axes = axes.clone();
                            let view = views[&op_id].clone();
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if axes.contains(&(a as UAxis)) {
                                    // Reverse the axis: input coord = len - 1 - out_idx.
                                    let len_m1 = self.sub(d.len, one);
                                    let idx = self.sub(len_m1, d.idx);
                                    new_view.push(SDim::new(idx, d.len));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                        &MoveOp::Pad { axis, lp, len } => {
                            // Pure backward pad (tinygrad): the input coordinate is
                            // the output coordinate shifted left by `lp` (a negative
                            // `lp` is a slice, shifting right), and the input extent
                            // is `len - lp - rp`, with `rp = len - lp - orig_len`
                            // recovered from x's own axis length. The resulting
                            // `idx >= 0 && idx < len` bounds check at the load is the
                            // exact validity mask -- no separate pad terms.
                            let mut view = views[&op_id].clone();
                            let d = view[axis].clone();
                            let idx = self.sub(d.idx, lp);
                            let orig = {
                                let dims = self.shape_ids(x);
                                dims[axis as usize]
                            };
                            let rp = self.sub(len, lp);
                            let rp = self.sub(rp, orig);
                            let in_len = self.sub(d.len, lp);
                            let in_len = self.sub(in_len, rp);
                            view[axis] = SDim::new(idx, in_len);
                            views.insert(x, view);
                        }
                        &MoveOp::Narrow { axis, start, .. } => {
                            let x_shape = self.shape_ids(x);
                            let view = views[&op_id].clone();
                            // Pure backward narrow: the input coordinate along the
                            // narrowed axis is `start + out_idx`, and the axis length
                            // is the input's own length on that axis. Other axes pass
                            // through unchanged.
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if a as UAxis == axis {
                                    let idx = self.add(d.idx, start);
                                    new_view.push(SDim::new(idx, x_shape[a]));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    if let Some(view) = views.get(&op_id).cloned() {
                        views.insert(x, view);
                    }
                }
                Op::Binary { x, y, .. } => {
                    if let Some(view) = views.get(&op_id).cloned() {
                        views.insert(x, view.clone());
                        views.insert(y, view);
                    }
                }
                Op::Index { .. } => {}
                Op::Stack { ref ops } => {
                    // Stack produces `[n] + first_shape`: output element at leading
                    // index `i` and trailing indices `t` reads input `i` at `t`. So
                    // each input reads at the trailing axes of the output view (the
                    // leading axis selects which source). Assign each stacked input
                    // the output view with the leading SDim dropped, then resolve the
                    // op into a chain of branchless selects on the leading group
                    // index: where(lead==n-1, src_{n-1}, ... where(lead==1, src_1,
                    // src_0)). The src_{k} are the input op ids; when the reverse
                    // walk reaches their Param they are remapped to loads and
                    // this chain follows automatically.
                    let stacked = ops.to_vec();
                    if let Some(view) = views.get(&op_id).cloned() {
                        debug_assert!(!view.is_empty(), "Stack: empty output view");
                        let leading = view[0].idx;
                        let trailing = &view[1..];
                        for &input in stacked.iter() {
                            views.insert(input, trailing.to_vec());
                        }
                        let n = stacked.len();
                        let mut ret = stacked[n - 1];
                        for k in (0..n - 1).rev() {
                            let k_const = self.push_back(Op::Const(Constant::idx(k as i64)));
                            let eq = self.eq(leading, k_const);
                            ret = self.branchless_where(eq, stacked[k], ret);
                        }
                        self.remap(op_id, ret);
                    }
                    self.remove_op(op_id);
                }
                ref op => {
                    self.debug();
                    unreachable!("{op:?}");
                }
            }
        }
    }

    fn toposort(&mut self, ro_params: &[OpId], rw_params: &[OpId]) {
        // Phase 2: collect reachable ops from the store roots, then topologically
        // order their dependencies. Phase 1 may leave the linked list temporarily
        // invalid while inserting and replacing ops, so the slab is the source of
        // truth until this phase rebuilds the list.
        let mut roots = Vec::new();
        for (op_id, op) in self.iter_unordered() {
            match op {
                Op::Store { .. } => roots.push(op_id),
                Op::Param { .. }
                | Op::Const(_)
                | Op::Binary { .. }
                | Op::Unary { .. }
                | Op::Cast { .. }
                | Op::Mad { .. }
                | Op::Load { .. }
                | Op::Index { .. }
                | Op::Reduce { .. }
                | Op::Loop { .. } => {}
                Op::Storage { .. } | Op::Wmma { .. } | Op::Barrier | Op::If { .. } | Op::EndIf | Op::EndLoop => {
                    debug_assert!(false, "unexpected root operation after Phase 1: {op:?}");
                }
                _ => {}
            }
        }

        // Reachability from the store roots. Any op not on a store's dependency
        // chain is dead and removed.
        let mut reachable = Set::default();
        let mut pending = roots;
        for _ in 0..50_000 {
            let Some(op_id) = pending.pop() else { break };
            if self.ops.contains_id(op_id) {
                if reachable.insert(op_id) {
                    pending.extend(self.at(op_id).parameters());
                }
            }
        }
        if !pending.is_empty() {
            panic!("toposort did not finish in 50000 steps");
        }

        for op_id in self.ops.ids().collect::<Vec<_>>() {
            if !reachable.contains(&op_id) && !matches!(self.ops[op_id].op, Op::Param { .. }) {
                self.remove_op(op_id);
            }
        }

        // Get reduce ids in sorted order, from innermost to outermost
        let mut reduce_ids: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        for _ in 0..50_000 {
            if op_id.is_null() {
                break;
            }
            if matches!(self.ops[op_id].op, Op::Reduce { .. }) {
                reduce_ids.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("toposort did not finish in 10000 steps");
        }

        // Structural edges between loops, derived purely from reduce
        // dependencies. If reduce `d` is a transitive dependency of reduce
        // `r` (d feeds r directly or indirectly), then `r` is the outer
        // region: its loop must open before `d`'s loop. Together with the
        // natural data dependency R_d -> ... -> R_r this makes partially
        // overlapping regions impossible.
        let mut extra_deps: Vec<(OpId, OpId)> = Vec::new(); // (producer, consumer)
        for &r in &reduce_ids {
            let Op::Reduce { x, reduce_axis: r_axis, .. } = self.ops[r].op else {
                unreachable!()
            };
            let mut stack: Vec<OpId> = vec![x];
            let mut seen: Set<OpId> = Set::default();
            for _ in 0..50_000 {
                let Some(p) = stack.pop() else { break };
                if p.is_null() || !seen.insert(p) {
                    continue;
                }
                if let Op::Reduce { reduce_axis: d_axis, .. } = self.ops[p].op {
                    extra_deps.push((r_axis, d_axis));
                }
                stack.extend(self.ops[p].op.parameters());
            }
            debug_assert!(stack.is_empty(), "dependency walk did not finish");
        }

        // Loop trip lengths, for sibling ordering (bigger loops first).
        let loop_size = |axis: OpId| -> Dim {
            let Op::Loop { len } = self.ops[axis].op else {
                unreachable!("reduce_axis must point at a Loop")
            };
            match self.ops[len].op {
                Op::Const(c) => c.as_dim().unwrap_or(0),
                _ => 0,
            }
        };

        // Region isolation: an independent reduce region (one whose input does
        // not feed another reduce and vice versa) must be FULLY SCHEDULED
        // before the next independent region's `Loop` header. Scope is assigned
        // positionally in `add_control_flow` — each `EndLoop` lands immediately
        // before its reduce op — so any unrelated op emitted between a region's
        // header and its reduce op would be silently captured into that region
        // and rejected by `kernel::verify` ("uses ... before declaration").
        // Ordering one region's RESULT before the other's HEADER suffices to
        // close this gap; the producer wins by bigger resolved trip length,
        // falling back to lower reduce op id for determinism.
        let mut siblings: Vec<(Dim, OpId, OpId)> = Vec::with_capacity(reduce_ids.len()); // (size, reduce_op, loop_op)
        for &r in &reduce_ids {
            let Op::Reduce { reduce_axis, .. } = self.ops[r].op else {
                unreachable!()
            };
            siblings.push((loop_size(reduce_axis), r, reduce_axis));
        }
        siblings.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut closures: Map<OpId, Set<OpId>> = Map::default();
        for &r in &reduce_ids {
            let Op::Reduce { x, .. } = self.ops[r].op else {
                unreachable!()
            };
            let mut stack: Vec<OpId> = vec![x];
            let mut seen = Set::default();
            for _ in 0..50_000 {
                let Some(p) = stack.pop() else { break };
                if p.is_null() || !seen.insert(p) {
                    continue;
                }
                stack.extend(self.ops[p].op.parameters());
            }
            debug_assert!(stack.is_empty(), "dependency walk did not finish");
            closures.insert(r, seen);
        }
        for w in siblings.windows(2) {
            let (_, a_red, _) = w[0];
            let (_, b_red, b_loop) = w[1];
            if !closures[&a_red].contains(&b_red) && !closures[&b_red].contains(&a_red) {
                extra_deps.push((a_red, b_loop));
                break;
            }
        }

        // ASAP Kahn: emit an op as soon as all its producers are placed,
        // preferring non-loops over loops (loops go last among ready ops,
        // so loop-invariant computation hoists above the loop headers)
        // and bigger loops before smaller ones among ready siblings.
        let mut in_degree: Map<OpId, u32> = Map::default();
        let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
        for (op_id, op) in self.iter_unordered() {
            if !reachable.contains(&op_id) {
                continue;
            }
            for p in op.parameters() {
                if !p.is_null() {
                    *in_degree.entry(op_id).or_default() += 1;
                    consumers.entry(p).or_default().push(op_id);
                }
            }
        }
        for &(prod, cons) in &extra_deps {
            *in_degree.entry(cons).or_default() += 1;
            consumers.entry(prod).or_default().push(cons);
        }

        let mut heap: BinaryHeap<std::cmp::Reverse<(u8, u64, OpId)>> = BinaryHeap::new();
        for &id in &reachable {
            if in_degree.get(&id).copied().unwrap_or(0) == 0 {
                let is_loop = matches!(self.ops[id].op, Op::Loop { .. });
                let size = if is_loop { loop_size(id) } else { 0 };
                heap.push(std::cmp::Reverse((u8::from(is_loop), u64::MAX - size as u64, id)));
            }
        }
        let mut order = Vec::with_capacity(reachable.len());
        for _ in 0..50_000 {
            let Some(std::cmp::Reverse((_, _, op_id))) = heap.pop() else {
                break;
            };
            order.push(op_id);
            if let Some(cs) = consumers.get(&op_id) {
                for &c in cs {
                    let d = in_degree.get_mut(&c).expect("consumer must have an in_degree entry");
                    *d -= 1;
                    if *d == 0 {
                        let is_loop = matches!(self.ops[c].op, Op::Loop { .. });
                        let size = if is_loop { loop_size(c) } else { 0 };
                        heap.push(std::cmp::Reverse((u8::from(is_loop), u64::MAX - size as u64, c)));
                    }
                }
            }
        }
        if order.len() != reachable.len() {
            panic!("linearize dependency ordering contains a cycle or missing operation");
        }
        order.retain(|op| !matches!(self.ops[*op].op, Op::Param { .. }));

        // Move the params to the front: read-only (Variable + Global) first,
        // then writable (GlobalMut), each in linked-list order. Unused
        // params never enter the Kahn order at all; used ones are dropped
        // here and reinserted.
        let mut final_order = Vec::with_capacity(order.len() + ro_params.len() + rw_params.len());
        final_order.extend(ro_params.iter().copied());
        final_order.extend(rw_params.iter().copied());
        final_order.extend(order);

        // Rebuild the kernel's linked list in `final_order`.
        for (i, &op) in final_order.iter().enumerate() {
            self.ops[op].prev = if i == 0 { OpId::NULL } else { final_order[i - 1] };
            self.ops[op].next = if i + 1 == final_order.len() {
                OpId::NULL
            } else {
                final_order[i + 1]
            };
        }
        self.head = final_order.first().copied().unwrap_or(OpId::NULL);
        self.tail = final_order.last().copied().unwrap_or(OpId::NULL);
    }

    // Auto-cast scalar operands in arithmetic so mixed-dtype binaries are
    // well-typed. Runs after toposort so it sees every Binary/Mad in the final
    // op list. Symmetric over all operands: if dtypes differ, every operand
    // with shape `[]` (constants, variables, and index/loop scalars) is cast so
    // the operation is well-typed; if no operand is scalar, the kernel is
    // broken and this panics.
    fn autocast_scalars(&mut self) {
        let ops: Vec<OpId> = {
            let mut v = Vec::new();
            let mut op_id = self.head;
            for _ in 0..50_000 {
                if op_id.is_null() {
                    break;
                }
                v.push(op_id);
                op_id = self.next_op(op_id);
            }
            if !op_id.is_null() {
                panic!("autocast_scalars did not finish in 10000 steps");
            }
            v
        };
        for op_id in ops {
            let operands: Vec<OpId> = match self.ops[op_id].op {
                Op::Binary { x, y, .. } => vec![x, y],
                Op::Mad { x, y, z } => vec![x, y, z],
                Op::Load { index, .. } | Op::Store { index, .. } => {
                    if index.is_null() || self.dtype(index) == IDX_T {
                        continue;
                    }
                    let cast = self.insert_before(op_id, Op::Cast { x: index, dtype: IDX_T });
                    match &mut self.ops[op_id].op {
                        Op::Load { index, .. } | Op::Store { index, .. } => *index = cast,
                        _ => unreachable!(),
                    }
                    continue;
                }
                _ => continue,
            };
            let dtypes: Vec<DType> = operands.iter().map(|&o| self.dtype(o)).collect();
            if dtypes.iter().all(|&d| d == dtypes[0]) {
                continue;
            }
            // Scalars are operands whose value shape is `[]`.
            let scalars: Vec<(OpId, DType)> =
                operands.iter().copied().zip(dtypes.iter().copied()).filter(|&(o, _)| self.shape(o).is_empty()).collect();
            if scalars.is_empty() {
                self.debug();
                panic!("autocast_scalars: mixed-dtype op {op_id} has no scalar operand to cast");
            }
            let target = if scalars.len() == operands.len() {
                // All operands are scalars: fold least_upper_dtype over the distinct dtypes.
                let mut target = dtypes[0];
                for &d in &dtypes[1..] {
                    target = target.least_upper_dtype(d);
                }
                target
            } else {
                // Mixed: every non-scalar operand must share one dtype; cast the
                // scalars to it.
                let nonscalar: Vec<DType> = operands
                    .iter()
                    .copied()
                    .zip(dtypes.iter().copied())
                    .filter(|&(o, _)| !self.shape(o).is_empty())
                    .map(|(_, d)| d)
                    .collect();
                let target = nonscalar[0];
                if nonscalar.iter().any(|&d| d != target) {
                    self.debug();
                    panic!("autocast_scalars: mixed-dtype op {op_id} has non-scalar operands of differing dtype");
                }
                target
            };
            let mut rewrites: Vec<(OpId, OpId)> = Vec::new();
            for &(o, d) in &scalars {
                if d != target {
                    rewrites.push((o, self.insert_before(op_id, Op::Cast { x: o, dtype: target })));
                }
            }
            if !rewrites.is_empty() {
                let map: Map<OpId, OpId> = rewrites.into_iter().collect();
                self.ops[op_id].op.remap_params(&map);
            }
        }
    }

    fn add_control_flow(&mut self) {
        // Phase 3: insert accumulators immediately before their exact loops.
        // No loop movement occurs after this point.
        let reduce_ids: Vec<OpId> =
            self.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).map(|(id, _)| id).collect();
        for op_id in reduce_ids {
            let Op::Reduce { x, rop, reduce_axis } = self.ops[op_id].op else {
                unreachable!()
            };
            let loop_id = reduce_axis;

            let acc_dtype = self.dtype(x);
            let zero = self.insert_const_idx_before(loop_id, 0u32);
            let acc_init = self.insert_before(
                loop_id,
                Op::Const(match rop {
                    BOp::Add => acc_dtype.zero_constant(),
                    BOp::Max => acc_dtype.min_constant(),
                    BOp::Mul => acc_dtype.one_constant(),
                    _ => unreachable!(),
                }),
            );
            let acc = self.insert_before(loop_id, Op::Storage { dtype: acc_dtype, scope: MemScope::Register, len: 1 });
            self.insert_before(loop_id, Op::Store { dst: acc, src: acc_init, index: zero, layout: MemLayout::Scalar });

            // Accumulate inside the loop, then close it, then read the result.
            let load_acc = self.insert_before(op_id, Op::Load { src: acc, index: zero, layout: MemLayout::Scalar });
            let bin_acc = self.insert_before(op_id, Op::Binary { x, y: load_acc, bop: rop });
            self.insert_before(op_id, Op::Store { dst: acc, src: bin_acc, index: zero, layout: MemLayout::Scalar });
            self.insert_before(op_id, Op::EndLoop);
            self.ops[op_id].op = Op::Load { src: acc, index: zero, layout: MemLayout::Scalar };
        }
    }
}
