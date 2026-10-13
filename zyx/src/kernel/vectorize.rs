// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::autotune::Optimization;
use crate::{
    Set,
    dtype::Constant,
    kernel::{Bindings, BOp, CoverGroup, Kernel, Op, OpId, Pat},
    shape::Dim,
};

/// Combine scalar loads into vectorized loads for better memory bandwidth.
#[derive(Debug)]
pub struct Vectorize {
    /// Supported vector lengths for this device.
    pub supported_lens: Vec<u8>,
    pub vectorize_ops: bool,
    /// Nesting level to vectorize (1 = flat vectors, 2 = vectors of
    /// vectors). Driven by `DeviceInfo::vector_depth`.
    pub level: u8,
}

impl Optimization for Vectorize {
    fn nconfigs(&self) -> u64 {
        1
    }

    fn apply(&self, kernel: &mut Kernel, _config: u64) {
        // Zero level gates: loads/stores iterate rank rounds
        // 1..=level internally; the rest are rank-general as written
        // and self-limit through supported_lens.
        kernel.vectorize_loads(&self.supported_lens, self.level);
        kernel.vectorize_stores(&self.supported_lens, self.level);
        if self.vectorize_ops {
            kernel.vectorize_ops_forward(&self.supported_lens);
            kernel.vectorize_ops_backward(&self.supported_lens);
            kernel.reduce_lanes(&self.supported_lens);
        }
    }
}

/// Compute-op head for one cover call of [`Kernel::vectorize_ops_forward`].
#[derive(Clone, Copy)]
enum FwdKind {
    Unary,
    Cast,
    Bitcast,
    Binary,
}

impl Kernel {
    /// Make the `Vectorize` optimization. Reads the bound device's
    /// supported vector widths and nesting depth itself.
    pub fn opt_vectorize(&self) -> Box<dyn Optimization> {
        let dev_info = self.dev_info();
        Box::new(Vectorize {
            supported_lens: dev_info.supported_vec_lens.clone(),
            vectorize_ops: !dev_info.supported_vec_lens.is_empty(),
            level: dev_info.vector_depth,
        })
    }

    /// Vectorize loads, general over rank.
    ///
    /// One round per level: round r groups loads of shape-rank r-1
    /// (scalars, then vectors, then tiles, ...) sharing one base with
    /// contiguous outer offsets into a single higher-rank load + Index
    /// lanes. Each round re-scans, so earlier rounds' loads group again
    /// deeper. `supported_lens` is the list of group sizes the target
    /// device supports (applies at every rank). Grouping is two cover
    /// matches: range-factored affine index (`Range*const + const`) and
    /// absolute const index.
    /// TODO for now this function ignores aliasing of stores and loads.
    pub fn vectorize_loads(&mut self, supported_lens: &[u8], level: u8) {
        for round in 1..=level {
            let rank = (round - 1) as usize;
            self.normalize_bare_lane0();
            let affine = Pat::load(
                Pat::bind('s'),
                Pat::all([Pat::bind('i'), Pat::binary(BOp::Add, Pat::bind('b'), Pat::bind_const('o'))]),
            );
            for g in self.match_cover(&affine, &['s', 'b'], supported_lens) {
                self.apply_load_cover(&g, rank);
            }
            let absolute = Pat::load(Pat::bind('s'), Pat::all([Pat::bind('i'), Pat::bind_const('o')]));
            for g in self.match_cover(&absolute, &['s'], supported_lens) {
                self.apply_load_cover(&g, rank);
            }
        }
    }

    /// Lane-0 members materialize as the bare range-factored base
    /// (their `Add(base, 0)` folded): rewrite them into the add form so
    /// one pattern covers every member.
    fn normalize_bare_lane0(&mut self) {
        let bare = Pat::load(
            Pat::bind('s'),
            Pat::all([Pat::bind('i'), Pat::binary(BOp::Mul, Pat::all([Pat::bind('b'), Pat::Range]), Pat::bind_const('k'))]),
        );
        let mut id = self.head;
        while !id.is_null() {
            if let Some(m) = self.match_pat(id, &bare) {
                let bytes = match self.at(id) {
                    Op::Load { bytes, .. } => *bytes,
                    _ => {
                        unreachable!("bare: load pattern root is a load")
                    }
                };
                let zero = self.insert_before(id, Op::Const(Constant::new::<i64>(0)));
                let norm = self.insert_before(id, Op::Binary { x: m.op('i'), y: zero, bop: BOp::Add });
                self.ops[id].op = Op::Load { src: m.op('s'), src_idx: norm, bytes };
            }
            id = self.next_op(id);
        }
    }

    /// Lane-sorted (lane, root, binding-index) of one cover group.
    fn cover_pairs(g: &CoverGroup) -> Vec<(i64, OpId, usize)> {
        let mut pairs: Vec<(i64, OpId, usize)> =
            g.roots.iter().copied().zip(g.lanes.iter().copied()).enumerate().map(|(k, (root, lane))| (lane, root, k)).collect();
        pairs.sort_by_key(|(lane, _, _)| *lane);
        pairs
    }

    /// One cover group of loads sharing (src, base): the roots' shape
    /// gates the round's rank and its footprint is the lane stride; the
    /// members get one higher-rank load (`[n] + elem` descriptor) and
    /// are rewritten to `Index` lanes in place. The anchor index is the
    /// smallest lane member's index. (`g.elem` shape agreement and lane
    /// progression are cover facts.)
    fn apply_load_cover(&mut self, g: &CoverGroup, rank: usize) {
        if g.elem.len() != rank || g.stride != g.elem.iter().product::<Dim>() || g.stride < 1 {
            return;
        }
        let pairs = Self::cover_pairs(g);
        let src = g.bindings.op('s');
        let base_root = pairs[0].1;
        let base_index = g.roots_bindings[pairs[0].2].op('i');
        let n = pairs.len();
        let cn = self
            .insert_before(base_root, Op::Const(Constant::new::<u16>(n.try_into().expect("vectorize: group overflows u16"))));
        let bytes = if g.elem.is_empty() {
            cn
        } else {
            let mut dims: Vec<OpId> = Vec::with_capacity(g.elem.len() + 1);
            dims.push(cn);
            for &d in &g.elem {
                let c = self.insert_before(
                    base_root,
                    Op::Const(Constant::new::<u16>(d.try_into().expect("vectorize: width overflows u16"))),
                );
                dims.push(c);
            }
            self.insert_before(base_root, Op::Stack { ops: dims.into_boxed_slice() })
        };
        let tload = self.insert_before(base_root, Op::Load { src, src_idx: base_index, bytes });
        for (row, (_, id, _)) in pairs.iter().enumerate() {
            self.ops[*id].op = Op::Index { vec: tload, idx: row };
        }
    }

    /// Vectorize stores, general over rank (mirror of
    /// [`Self::vectorize_loads`]): round r groups stored values of
    /// shape-rank r-1 sharing one base with contiguous outer offsets
    /// into a single higher-rank store, values stacked per lane. Stored
    /// values ride the per-root bindings.
    pub fn vectorize_stores(&mut self, supported_lens: &[u8], level: u8) {
        for round in 1..=level {
            let rank = (round - 1) as usize;
            let affine = Pat::store(
                Pat::bind('d'),
                Pat::all([Pat::bind('i'), Pat::binary(BOp::Add, Pat::bind('b'), Pat::bind_const('o'))]),
                Pat::bind('x'),
            );
            for g in self.match_cover(&affine, &['d', 'b'], supported_lens) {
                self.apply_store_cover(&g, rank);
            }
            let absolute = Pat::store(Pat::bind('d'), Pat::all([Pat::bind('i'), Pat::bind_const('o')]), Pat::bind('x'));
            for g in self.match_cover(&absolute, &['d'], supported_lens) {
                self.apply_store_cover(&g, rank);
            }
        }
    }

    /// One cover group of stores sharing (dst, base): roots' shape and
    /// lane stride per the mirror. Values come off the per-root
    /// bindings; the stack goes after the last member store so all
    /// values are declared.
    fn apply_store_cover(&mut self, g: &CoverGroup, rank: usize) {
        if g.elem.len() != rank || g.stride != g.elem.iter().product::<Dim>() || g.stride < 1 {
            return;
        }
        let pairs = Self::cover_pairs(g);
        let dst = g.bindings.op('d');
        let n = pairs.len();
        // Values stacked at the lane order; the stack goes after the
        // last member so all values are declared.
        let mut ops = vec![OpId::NULL; n].into_boxed_slice();
        for (row, (_, _, k)) in pairs.iter().enumerate() {
            ops[row] = g.roots_bindings[*k].op('x');
        }
        let mut last_id = pairs[0].1;
        let mut cur = pairs[0].1;
        while !cur.is_null() {
            if pairs.iter().any(|(_, id, _)| *id == cur) {
                last_id = cur;
            }
            cur = self.next_op(cur);
        }
        let vstore = self.insert_after(last_id, Op::Stack { ops });
        let mut prev = vstore;
        prev = self.insert_after(prev, Op::Const(Constant::new::<u16>(n.try_into().expect("vectorize: group overflows u16"))));
        let bytes = if g.elem.is_empty() {
            prev
        } else {
            let cn = prev;
            let mut dims: Vec<OpId> = Vec::with_capacity(g.elem.len() + 1);
            dims.push(cn);
            for &d in &g.elem {
                let c = self.insert_after(prev, Op::Const(Constant::new::<u16>(d.try_into().expect("vectorize: width overflows u16"))));
                dims.push(c);
                prev = c;
            }
            self.insert_after(prev, Op::Stack { ops: dims.into_boxed_slice() })
        };
        let base_index = g.roots_bindings[pairs[0].2].op('i');
        self.insert_after(bytes, Op::Store { dst, dst_idx: base_index, src: vstore, bytes });
        for (_, id, _) in &pairs {
            self.remove_op(*id);
        }
    }

    /// Check whether `target` appears after `pos` in the IR op chain.
    fn op_is_before(&self, target: OpId, pos: OpId) -> bool {
        let mut cur = self.head;
        while !cur.is_null() && cur != pos {
            if cur == target {
                return true;
            }
            cur = self.next_op(cur);
        }
        false
    }

    /// Whether any op before `stop` (exclusive) in kernel order uses `op`.
    fn use_before(&self, op: OpId, stop: OpId) -> bool {
        let mut cur = self.head;
        while !cur.is_null() && cur != stop {
            if self.ops[cur].op.parameters().any(|p| p == op) {
                return true;
            }
            cur = self.next_op(cur);
        }
        false
    }

    /// Return the op from `candidates` that appears last in the linked list.
    fn last_in_kernel_order(&self, candidates: &[OpId]) -> OpId {
        let set: Set<OpId> = candidates.iter().copied().collect();
        let mut cur = self.tail;
        while !cur.is_null() {
            if set.contains(&cur) {
                return cur;
            }
            cur = self.prev_op(cur);
        }
        candidates[0]
    }

    /// Walk forward, find scalar ops over `Index` lanes of one vector
    /// that share the same op head, and bundle them: `Stack` the
    /// operands per position, rebuild the same op over the vectors,
    /// devectorize results. One cover call per head kind and lane
    /// position (head agreement is the payload binder); partial takes
    /// (largest supported count, leftovers stay scalar) ride the fuel
    /// loop.
    pub fn vectorize_ops_forward(&mut self, supported_lens: &[u8]) {
        // Process in descending length order for maximal vectorization first
        let mut supported: Vec<u8> = supported_lens.to_vec();
        supported.sort_unstable_by(|a, b| b.cmp(a));
        // The lane operand identity is IN the pattern: `same vec, same
        // lane, same head` — the binary retry covers both operand
        // orders and the use slots come off the match ('n' binds the
        // lane Index node itself).
        let lane = Pat::index_node('n', Pat::bind('v'), 'k');
        let pats = [
            (Pat::unary(lane.clone(), 'p'), FwdKind::Unary, 1usize),
            (Pat::cast(lane.clone(), 'p'), FwdKind::Cast, 1usize),
            (Pat::bitcast(lane.clone(), 'p'), FwdKind::Bitcast, 1usize),
            (Pat::binary('p', lane, Pat::bind('y')), FwdKind::Binary, 2usize),
        ];
        // Fuel-capped: each iteration must vectorize one group; the cap
        // turns a no-progress bug into termination, never a hang.
        for _ in 0..10000 {
            let mut applied = false;
            for (pat, kind, arity) in &pats {
                if self.vectorize_ops_cover(pat, *kind, *arity, &supported) {
                    applied = true;
                    break;
                }
            }
            if !applied {
                break;
            }
        }
    }

    /// One cover group of same-head compute ops over lanes of one
    /// vector: take the largest supported prefix, stack operands per
    /// position, rebuild the head over the vectors, devectorize the
    /// take. The lane use slot per member comes off the match
    /// (`slots('k')`). Returns whether one group applied.
    fn vectorize_ops_cover(&mut self, pat: &Pat, kind: FwdKind, arity: usize, supported: &[u8]) -> bool {
        for g in self.match_cover(pat, &['v', 'p'], supported) {
            let Some(selected) = g.take_prefix(supported) else {
                continue;
            };
            let v = g.bindings.op('v');
            let first = selected[0];
            // Per-member lane slot and lane op, off the match.
            let lane_of = |k: OpId, m: &Bindings| -> OpId {
                self.ops[k].op.parameters().nth(*m.slots('n').first().expect("lane slot")).expect("arity drift")
            };
            let members: Vec<(OpId, &Bindings)> = selected.iter().copied().zip(g.roots_bindings.iter()).collect();
            // Every Stack operand must be declared before `first`.
            // Unrolled bodies interleave lanes with consumers (load,
            // cast, load, cast, ...), leaving later lanes after `first`.
            // Hoist those lanes to just before `first`: a lane depends
            // only on the source vec, so the move is declaration-safe
            // iff the source vec precedes `first` and no hoisted lane
            // is used before `first`. Otherwise the group stays scalar
            // for the store-driven backward pass, as before.
            // (Multi-operand ops only guard the other side; lanes are
            // assumed in place, matching prior behavior.)
            if arity == 1 {
                let late: Vec<OpId> =
                    members.iter().map(|(c, m)| lane_of(*c, m)).filter(|&l| !self.op_is_before(l, first)).collect();
                if !late.is_empty() {
                    let mut hoist_ok = self.op_is_before(v, first);
                    for l in &late {
                        if self.use_before(*l, first) {
                            hoist_ok = false;
                            break;
                        }
                    }
                    if !hoist_ok {
                        continue;
                    }
                    for l in late {
                        self.move_op_before(l, first);
                    }
                }
            } else {
                let mut other_ok = true;
                for (c, m) in &members {
                    let slot = *m.slots('n').first().expect("lane slot");
                    let operands: Vec<OpId> = self.ops[*c].op.parameters().collect();
                    for (i, &o) in operands.iter().enumerate() {
                        if i != slot && !self.op_is_before(o, first) {
                            other_ok = false;
                            break;
                        }
                    }
                    if !other_ok {
                        break;
                    }
                }
                if !other_ok {
                    continue;
                }
            }

            // Apply: Stack each operand position across members, rebuild
            // the head over the vectors.
            let mat: Vec<Vec<OpId>> = selected.iter().map(|&c| self.ops[c].op.parameters().collect()).collect();
            let mut vec_ops = Vec::with_capacity(arity);
            for i in 0..arity {
                let stack: Box<[OpId]> = mat.iter().map(|row| row[i]).collect();
                vec_ops.push(self.insert_before(first, Op::Stack { ops: stack }));
            }
            let mut rebuilt = match kind {
                FwdKind::Unary => Op::Unary { x: OpId::NULL, uop: g.bindings.uop('p') },
                FwdKind::Cast => Op::Cast { x: OpId::NULL, dtype: g.bindings.dtype('p') },
                FwdKind::Bitcast => Op::Bitcast { x: OpId::NULL, dtype: g.bindings.dtype('p') },
                FwdKind::Binary => Op::Binary { x: OpId::NULL, y: OpId::NULL, bop: g.bindings.bop('p') },
            };
            for (p, vv) in rebuilt.parameters_mut().zip(vec_ops) {
                *p = vv;
            }
            let vec_op_id = self.insert_before(first, rebuilt);
            for (i, &consumer) in selected.iter().enumerate() {
                self.ops[consumer].op = Op::Index { vec: vec_op_id, idx: i };
            }
            return true;
        }
        false
    }

    /// Walk backward, find `Stack[X0..Xn]` where all Xi are the same
    /// compute op (head agreement via [`Kernel::match_stack`]), and
    /// collapse it: compute_op(stack[inputs of Xi]) replaces the stack.
    pub fn vectorize_ops_backward(&mut self, supported_lens: &[u8]) {
        let mut op_id = self.tail;
        while !op_id.is_null() {
            let next = self.prev_op(op_id);
            let is_stack = matches!(self.ops[op_id].op, Op::Stack { .. });
            if is_stack {
                self.apply_stack_collapse(op_id, supported_lens);
            }
            op_id = next;
        }
    }

    /// Collapse one stack whose members share a compute-op head. Each
    /// head kind gets a leaf pattern; `match_stack` returns per-member
    /// operand bindings. Insert new ops after the last source in kernel
    /// order, remap the stack to the new op.
    fn apply_stack_collapse(&mut self, stack: OpId, supported_lens: &[u8]) {
        let first = match &self.ops[stack].op {
            Op::Stack { ops } => ops.first().copied(),
            _ => {
                return;
            }
        };
        let Some(first) = first else {
            return;
        };
        // Leaf per head kind: operand binders are per-member op
        // bindings, the payload binder makes match_stack assert the
        // same-head condition across members. Arity records how many
        // operand positions the head carries.
        let (leaf, arity) = match self.at(first) {
            Op::Unary { .. } => (Pat::unary(Pat::bind('x'), 'p'), 1usize),
            Op::Cast { .. } => (Pat::cast(Pat::bind('x'), 'p'), 1usize),
            Op::Bitcast { .. } => (Pat::bitcast(Pat::bind('x'), 'p'), 1usize),
            Op::Binary { .. } => (Pat::binary('p', Pat::bind('x'), Pat::bind('y')), 2usize),
            _ => {
                return;
            }
        };
        let Some(members) = self.match_stack(stack, &leaf, supported_lens) else {
            return;
        };
        // Per-position sources; guard: no sub-op is also a source
        // (would create use-before-declaration in the IR).
        let mut positions: Vec<Vec<OpId>> = vec![Vec::with_capacity(members.len()); arity];
        let names = ['x', 'y'];
        for m in &members {
            for (p, name) in names.iter().enumerate().take(arity) {
                positions[p].push(m.op(*name));
            }
        }
        let all: Vec<OpId> = positions.iter().flatten().copied().collect();
        if all.iter().any(|s| matches!(&self.ops[stack].op, Op::Stack { ops } if ops.contains(s))) {
            return;
        }
        // Stack each position, rebuild the head over the stacks, remap.
        let last = self.last_in_kernel_order(&all);
        let mut stacks: Vec<OpId> = Vec::with_capacity(arity);
        let mut prev = last;
        for pos in &positions {
            let s = self.insert_after(prev, Op::Stack { ops: pos.clone().into_boxed_slice() });
            stacks.push(s);
            prev = s;
        }
        let mut rebuilt = match self.at(first) {
            Op::Unary { .. } => Op::Unary { x: OpId::NULL, uop: members[0].uop('p') },
            Op::Cast { .. } => Op::Cast { x: OpId::NULL, dtype: members[0].dtype('p') },
            Op::Bitcast { .. } => Op::Bitcast { x: OpId::NULL, dtype: members[0].dtype('p') },
            Op::Binary { .. } => Op::Binary { x: OpId::NULL, y: OpId::NULL, bop: members[0].bop('p') },
            _ => {
                unreachable!("arity came from this head")
            }
        };
        for (p, s) in rebuilt.parameters_mut().zip(stacks) {
            *p = s;
        }
        let v_op = self.insert_after(prev, rebuilt);
        self.remap(stack, v_op);
    }

    /// Fold an add-chain covering every lane of one vector into a scalar
    /// `VecReduce`. The whole reduction is one [`Pat::seq`]: a match IS
    /// the reduction, so this only walks roots and rewrites them in
    /// place; orphaned links die in DCE.
    pub fn reduce_lanes(&mut self, supported_lens: &[u8]) {
        let pat = Pat::seq(BOp::Add, Pat::index(Pat::bind('v'), 'k'), supported_lens);
        // Fuel-capped: one chain per iteration; the cap turns a
        // no-progress bug into termination, never a hang.
        for _ in 0..10000 {
            let mut found = None;
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Some(m) = self.match_pat(op_id, &pat) {
                    found = Some((op_id, m.op('v')));
                    break;
                }
                op_id = self.next_op(op_id);
            }
            let Some((root, vec)) = found else { break };
            self.ops[root].op = Op::VecReduce { x: vec, bop: BOp::Add };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DType,
        kernel::{BOp, Dev, Kernel, Op, UOp},
    };

    // Helper to verify c0/c1 were replaced with devecs, find vectorize + vector op
    fn check_forward_result(k: &Kernel, c0: crate::kernel::OpId, c1: crate::kernel::OpId) {
        assert!(matches!(k.ops[c0].op, Op::Index { .. }));
        assert!(matches!(k.ops[c1].op, Op::Index { .. }));
        let mut found_v = false;
        let mut found_vop = false;
        let mut op_id = k.head;
        while !op_id.is_null() {
            match &k.ops[op_id].op {
                Op::Stack { ops } if ops.len() == 2 => found_v = true,
                Op::Unary { uop: UOp::Cos, x } if matches!(k.ops[*x].op, Op::Stack { .. }) => found_vop = true,
                _ => {}
            }
            op_id = k.next_op(op_id);
        }
        assert!(found_v, "Vectorize not found");
        assert!(found_vop, "Vector cos not found");
    }

    #[test]
    fn vectorize_ops_forward_2_lane() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 2);
        let [s0, s1] = k.devectorize::<2>(vec_load);
        let c0 = k.cos(s0);
        let c1 = k.cos(s1);
        k.store(dst, c0, g0);
        let four = k.const_idx(4u32);
        let idx_c1 = k.add(g0, four);
        k.store(dst, c1, idx_c1);

        k.vectorize_ops_forward(&[2]);

        assert!(matches!(k.ops[s0].op, Op::Index { .. }));
        assert!(matches!(k.ops[s1].op, Op::Index { .. }));
        check_forward_result(&k, c0, c1);
    }

    #[test]
    fn vectorize_ops_forward_4_lane() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 4);
        let [s0, s1, s2, s3] = k.devectorize::<4>(vec_load);
        let [c0, c1, c2, c3] = [s0, s1, s2, s3].map(|s| k.cos(s));
        let c1i = k.const_idx(1u32);
        let c2i = k.const_idx(2u32);
        let c3i = k.const_idx(3u32);
        let i1 = k.add(g0, c1i);
        let i2 = k.add(g0, c2i);
        let i3 = k.add(g0, c3i);
        k.store(dst, c0, g0);
        k.store(dst, c1, i1);
        k.store(dst, c2, i2);
        k.store(dst, c3, i3);

        k.vectorize_ops_forward(&[2, 4]);

        // All 4 should be devecs after one pass (vectorized as length 4)
        assert!(matches!(k.ops[s0].op, Op::Index { .. }));
        assert!(matches!(k.ops[c0].op, Op::Index { .. }));
        assert!(matches!(k.ops[c1].op, Op::Index { .. }));
        assert!(matches!(k.ops[c2].op, Op::Index { .. }));
        assert!(matches!(k.ops[c3].op, Op::Index { .. }));

        let mut found_v = false;
        let mut found_vop = false;
        let mut found_cos_scalar = false;
        let mut op_id = k.head;
        while !op_id.is_null() {
            match &k.ops[op_id].op {
                Op::Stack { ops } if ops.len() == 4 => found_v = true,
                Op::Unary { uop: UOp::Cos, x } if matches!(k.ops[*x].op, Op::Stack { .. }) => found_vop = true,
                Op::Unary { uop: UOp::Cos, .. } => found_cos_scalar = true,
                _ => {}
            }
            op_id = k.next_op(op_id);
        }
        assert!(found_v, "Vectorize(4) not found");
        assert!(found_vop, "Vector cos(4) not found");
        assert!(!found_cos_scalar, "No scalar cos should remain");
    }

    #[test]
    fn vectorize_ops_forward_mixed_ops() {
        // 4 devecs, but only 2 have cos, other 2 have sin
        // After first pass: cos(2) is vectorized
        // After second pass: sin(2) is vectorized
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 4);
        let [s0, s1, s2, s3] = k.devectorize::<4>(vec_load);
        let c0 = k.cos(s0);
        let c1 = k.sin(s1);
        let c2 = k.cos(s2);
        let c3 = k.sin(s3);
        k.store(dst, c0, g0);
        let c1i = k.const_idx(1u32);
        let c2i = k.const_idx(2u32);
        let c3i = k.const_idx(3u32);
        let i1 = k.add(g0, c1i);
        let i2 = k.add(g0, c2i);
        let i3 = k.add(g0, c3i);
        k.store(dst, c0, g0);
        k.store(dst, c1, i1);
        k.store(dst, c2, i2);
        k.store(dst, c3, i3);

        k.vectorize_ops_forward(&[2, 4]);

        // After full run: all 4 consumers should be devecs
        // (cos was vectorized first as length-2, then sin as length-2)
        assert!(matches!(k.ops[c0].op, Op::Index { .. }));
        assert!(matches!(k.ops[c1].op, Op::Index { .. }));
        assert!(matches!(k.ops[c2].op, Op::Index { .. }));
        assert!(matches!(k.ops[c3].op, Op::Index { .. }));

        // Should have two Vectorize ops (cos group and sin group)
        let mut v_count = 0;
        let mut vop_count = 0;
        let mut op_id = k.head;
        while !op_id.is_null() {
            match &k.ops[op_id].op {
                Op::Stack { .. } => v_count += 1,
                Op::Unary { uop: UOp::Cos, x } if matches!(k.ops[*x].op, Op::Stack { .. }) => vop_count += 1,
                Op::Unary { uop: UOp::Sin, x } if matches!(k.ops[*x].op, Op::Stack { .. }) => vop_count += 1,
                _ => {}
            }
            op_id = k.next_op(op_id);
        }
        assert_eq!(v_count, 2, "Two Vectorize ops expected");
        assert_eq!(vop_count, 2, "Two vector Unary ops expected");
    }

    #[test]
    fn vectorize_ops_forward_binary() {
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 2);
        let [s0, s1] = k.devectorize::<2>(vec_load);
        let c = k.const_val(1.0f32);
        let r0 = k.add(s0, c);
        let r1 = k.add(s1, c);
        let four = k.const_idx(4u32);
        let idx1 = k.add(g0, four);
        k.store(dst, r0, g0);
        k.store(dst, r1, idx1);

        k.vectorize_ops_forward(&[2]);

        assert!(matches!(k.ops[s0].op, Op::Index { .. }));
        assert!(matches!(k.ops[s1].op, Op::Index { .. }));
        assert!(matches!(k.ops[r0].op, Op::Index { .. }));
        assert!(matches!(k.ops[r1].op, Op::Index { .. }));

        // Find the vector Binary op
        let mut found_vec_bin = false;
        let mut op_id = k.head;
        while !op_id.is_null() {
            if let Op::Binary { bop: BOp::Add, x, y } = &k.ops[op_id].op
                && matches!(k.ops[*x].op, Op::Stack { .. })
                && matches!(k.ops[*y].op, Op::Stack { .. })
            {
                found_vec_bin = true;
            }
            op_id = k.next_op(op_id);
        }
        assert!(found_vec_bin, "Vector Binary(Add) op not found");
    }

    #[test]
    fn vectorize_ops_forward_binary_y_pos() {
        // devec in Y position: c + devec(v, i)
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 2);
        let [s0, s1] = k.devectorize::<2>(vec_load);
        let c = k.const_val(1.0f32);
        let r0 = k.add(c, s0); // devec in Y
        let r1 = k.add(c, s1); // devec in Y
        let four = k.const_idx(4u32);
        let idx1 = k.add(g0, four);
        k.store(dst, r0, g0);
        k.store(dst, r1, idx1);

        k.vectorize_ops_forward(&[2]);

        assert!(matches!(k.ops[r0].op, Op::Index { .. }));
        assert!(matches!(k.ops[r1].op, Op::Index { .. }));
    }

    #[test]
    fn vectorize_loads_level2_forms_tile() {
        // Sum-like IR: 32 vec32 loads at outer offsets 0, 32, ... .
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let mut vloads = Vec::new();
        for r in 0..32u32 {
            let off = k.const_idx(r * 32);
            vloads.push(k.load_vector(src, off, 32));
        }

        k.vectorize_loads(&[32], 2);

        // One tile load, bytes = Stack[32, 32].
        let mut tiles = 0;
        let mut tload = crate::kernel::OpId::NULL;
        let mut op_id = k.head;
        while !op_id.is_null() {
            if let Op::Load { bytes, .. } = k.ops[op_id].op
                && k.shape(op_id) == vec![32, 32]
            {
                tiles += 1;
                tload = op_id;
                assert!(matches!(k.ops[bytes].op, Op::Stack { .. }));
            }
            op_id = k.next_op(op_id);
        }
        assert_eq!(tiles, 1, "one tile load expected");
        // Every original vload is now a row lane 0..32.
        let mut rows = vec![false; 32];
        for &v in &vloads {
            match k.ops[v].op {
                Op::Index { vec, idx } => {
                    assert_eq!(vec, tload);
                    rows[idx] = true;
                    assert_eq!(k.shape(v), vec![32]);
                }
                ref op => panic!("vload {v} not a row lane: {op:?}"),
            }
        }
        assert!(rows.iter().all(|&r| r), "rows 0..32 all covered");
    }

    #[test]
    fn vectorize_loads_range_factored_forms_tile() {
        // Post-coarsen pipeline shape: 32 vec32 loads with
        // idx = group_range * 32 + rowconst (range-factored, shared
        // range op across rows).
        let mut k = Kernel::new(Dev::Auto);
        let src = k.param(DType::F32);
        let num_rows = k.const_idx(32);
        let gr = k.group_range(0, num_rows);
        let b32 = k.const_idx(32);
        let base = k.mul(gr, b32);
        let mut vloads = Vec::new();
        for r in 0..32u32 {
            let off = k.const_idx(r * 32);
            let idx = k.add(base, off);
            vloads.push(k.load_vector(src, idx, 32));
        }

        k.vectorize_loads(&[32], 2);

        let mut tiles = 0;
        let mut tload = crate::kernel::OpId::NULL;
        let mut op_id = k.head;
        while !op_id.is_null() {
            if let Op::Load { .. } = k.ops[op_id].op {
                if k.shape(op_id) == vec![32, 32] {
                    tiles += 1;
                    tload = op_id;
                }
            }
            op_id = k.next_op(op_id);
        }
        assert_eq!(tiles, 1, "one tile load expected");
        let mut rows = vec![false; 32];
        for &v in &vloads {
            match k.ops[v].op {
                Op::Index { vec, idx } => {
                    assert_eq!(vec, tload);
                    rows[idx] = true;
                }
                ref op => panic!("vload {v} not a row lane: {op:?}"),
            }
        }
        assert!(rows.iter().all(|&r| r), "rows 0..32 all covered");
    }

    #[test]
    fn vectorize_ops_and_constfold_clears_vectorize_devectorize() {

        let mut k = Kernel::new(Dev::Auto);

        let src = k.param(DType::F32);
        let dst = k.param(DType::F32);
        let g0_len = k.const_idx(4);
        let g0 = k.group_range(0, g0_len);
        let two = k.const_idx(2u32);
        let offset = k.bit_shift_left(g0, two);
        let vec_load = k.load_vector(src, offset, 4);
        let [s0, s1, s2, s3] = k.devectorize::<4>(vec_load);
        let c0 = k.cos(s0);
        let c1 = k.cos(s1);
        let c2 = k.cos(s2);
        let c3 = k.cos(s3);
        let vec = k.stack(&[c0, c1, c2, c3]);
        k.store_vector(dst, vec, offset, 4);

        k.vectorize_ops_backward(&[4]);
        k.constant_folding();
        k.dead_code_elimination();

        let mut op_id = k.head;
        while !op_id.is_null() {
            match k.ops[op_id].op {
                Op::Stack { .. } | Op::Index { .. } => {
                    panic!("Found Vectorize/Devectorize op at {op_id} after passes");
                }
                _ => {}
            }
            op_id = k.next_op(op_id);
        }
    }
}
