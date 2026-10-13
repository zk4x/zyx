// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Pat},
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

#[derive(Debug)]
struct LoadInfo {
    id: OpId,
    index: OpId,
}

#[derive(Debug)]
struct StoreInfo {
    id: OpId,
    index: OpId,
    x: OpId,
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
    /// device supports (applies at every rank).
    /// TODO for now this function ignores aliasing of stores and loads.
    pub fn vectorize_loads(&mut self, supported_lens: &[u8], level: u8) {
        for round in 1..=level {
            let rank = (round - 1) as usize;
            let mut op_id = self.head;
            // Map: (src, elem width-shape) -> LoadInfo
            let mut loads: Vec<Map<(OpId, Vec<Dim>), Vec<LoadInfo>>> = Vec::new();
            loads.push(Map::default());
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Loop { .. } => {
                        loads.push(Map::default());
                    }
                    Op::Load { src, src_idx: index, bytes } => {
                        let elem: Option<Vec<Dim>> = if bytes.is_null() {
                            Some(Vec::new())
                        } else if let Op::Stack { ref ops } = self.ops[bytes].op {
                            ops.iter().map(|&d| self.resolve_const(d).and_then(Constant::as_dim)).collect()
                        } else {
                            self.resolve_const(bytes).and_then(Constant::as_dim).map(|w| vec![w])
                        };
                        if let Some(elem) = elem
                            && elem.len() == rank
                        {
                            loads
                                .last_mut()
                                .unwrap()
                                .entry((src, elem))
                                .or_default()
                                .push(LoadInfo { id: op_id, index });
                        }
                    }
                    Op::EndLoop => {
                        let scope = loads.pop().expect("vectorize: scope stack underflow");
                        self.verify_and_apply_vectorization(scope, supported_lens);
                    }
                    _ => {}
                }

                op_id = self.next_op(op_id);
            }

            let scope = loads.pop().expect("vectorize: scope stack underflow");
            self.verify_and_apply_vectorization(scope, supported_lens);
        }
    }

    /// Group one scope's loads by (src, elem shape): same width, same
    /// shared base, outer offsets covering exactly `{0, step, ...}` get
    /// one higher-rank load (`[n] + elem` descriptor) with every member
    /// rewritten to an `Index` lane in place. The caller pops one loop
    /// scope and passes it.
    fn verify_and_apply_vectorization(
        &mut self,
        loads: Map<(OpId, Vec<Dim>), Vec<LoadInfo>>,
        supported_lens: &[u8],
    ) {
        for ((src, elem), loads) in loads {
            let n = loads.len();
            if n >= 256 || !supported_lens.contains(&(n as u8)) {
                continue;
            }
            // Offset step = one element's footprint; degenerate
            // (zero/negative) widths never group.
            let step: Dim = elem.iter().product();
            if step < 1 {
                continue;
            }
            // Per-member const offset (consts fold into the NULL
            // stride) + shared base pattern (everything else must
            // match exactly across members).
            let mut offs: Vec<(OpId, OpId, Dim)> = Vec::with_capacity(n);
            let mut prev: Option<Vec<(OpId, (Dim, Dim))>> = None;
            let mut ok = true;
            for load in &loads {
                let Some(strides) = self.get_strides(load.index) else {
                    ok = false;
                    break;
                };
                let off = strides.get(&OpId::NULL).map(|(_, v)| *v).unwrap_or(0);
                let mut base: Vec<(OpId, (Dim, Dim))> =
                    strides.iter().filter(|(id, _)| !id.is_null()).map(|(id, v)| (*id, *v)).collect();
                base.sort();
                if prev.as_ref().is_some_and(|p| *p != base) {
                    ok = false;
                    break;
                }
                prev = Some(base);
                offs.push((load.id, load.index, off));
            }
            if !ok {
                continue;
            }
            offs.sort_by_key(|(_, _, off)| *off);
            if offs.iter().map(|(_, _, off)| *off).ne((0..(n as Dim)).map(|i| i * step)) {
                continue;
            }
            // Form the higher-rank load before the offset-0 member;
            // every member becomes its lane.
            let (base_id, base_index, _) = offs[0];
            let cn = self.insert_before(base_id, Op::Const(Constant::new::<u16>(n.try_into().expect("vectorize: group overflows u16"))));
            let bytes = if elem.is_empty() {
                cn
            } else {
                let mut ops: Vec<OpId> = Vec::with_capacity(elem.len() + 1);
                ops.push(cn);
                for &d in &elem {
                    ops.push(self.insert_before(
                        base_id,
                        Op::Const(Constant::new::<u16>(d.try_into().expect("vectorize: width overflows u16"))),
                    ));
                }
                self.insert_before(base_id, Op::Stack { ops: ops.into_boxed_slice() })
            };
            let tload = self.insert_before(base_id, Op::Load { src, src_idx: base_index, bytes });
            for (row, (id, _, _)) in offs.iter().enumerate() {
                self.ops[*id].op = Op::Index { vec: tload, idx: row };
            }
        }
    }

    /// Vectorize stores, general over rank (mirror of
    /// [`Self::vectorize_loads`]): round r groups stored values of
    /// shape-rank r-1 sharing one base with contiguous outer offsets
    /// into a single higher-rank store. `supported_lens` is the list
    /// of group sizes the target device supports (applies at every
    /// rank).
    pub fn vectorize_stores(&mut self, supported_lens: &[u8], level: u8) {
        for round in 1..=level {
            let rank = (round - 1) as usize;
            let mut op_id = self.head;
            // Map: (dst, elem width-shape) -> StoreInfo
            let mut stores: Vec<Map<(OpId, Vec<Dim>), Vec<StoreInfo>>> = Vec::new();
            stores.push(Map::default());
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Loop { .. } => {
                        stores.push(Map::default());
                    }
                    Op::Store { dst, dst_idx: index, src: x, bytes } => {
                        let elem: Option<Vec<Dim>> = if bytes.is_null() {
                            Some(Vec::new())
                        } else if let Op::Stack { ref ops } = self.ops[bytes].op {
                            ops.iter().map(|&d| self.resolve_const(d).and_then(Constant::as_dim)).collect()
                        } else {
                            self.resolve_const(bytes).and_then(Constant::as_dim).map(|w| vec![w])
                        };
                        if let Some(elem) = elem
                            && elem.len() == rank
                        {
                            stores
                                .last_mut()
                                .unwrap()
                                .entry((dst, elem))
                                .or_default()
                                .push(StoreInfo { id: op_id, index, x });
                        }
                    }
                    Op::EndLoop => {
                        let scope = stores.pop().expect("vectorize: scope stack underflow");
                        self.verify_and_apply_store_vectorization(scope, supported_lens);
                    }
                    _ => {}
                }

                op_id = self.next_op(op_id);
            }

            let scope = stores.pop().expect("vectorize: scope stack underflow");
            self.verify_and_apply_store_vectorization(scope, supported_lens);
        }
    }

    /// Group one scope's stores by (dst, elem shape): same width, same
    /// shared base, outer offsets covering exactly `{0, step, ...}`
    /// get one higher-rank store (`[n] + elem` descriptor) with the
    /// values stacked per offset. Stored values must already be
    /// declared: the stack goes after the last member store. The
    /// caller pops one loop scope and passes it.
    fn verify_and_apply_store_vectorization(
        &mut self,
        stores: Map<(OpId, Vec<Dim>), Vec<StoreInfo>>,
        supported_lens: &[u8],
    ) {
        for ((dst, elem), stores) in stores {
            let n = stores.len();
            if n >= 256 || !supported_lens.contains(&(n as u8)) {
                continue;
            }
            let step: Dim = elem.iter().product();
            if step < 1 {
                continue;
            }
            let mut offs: Vec<(OpId, OpId, OpId, Dim)> = Vec::with_capacity(n);
            let mut prev: Option<Vec<(OpId, (Dim, Dim))>> = None;
            let mut ok = true;
            for store in &stores {
                let Some(strides) = self.get_strides(store.index) else {
                    ok = false;
                    break;
                };
                let off = strides.get(&OpId::NULL).map(|(_, v)| *v).unwrap_or(0);
                let mut base: Vec<(OpId, (Dim, Dim))> =
                    strides.iter().filter(|(id, _)| !id.is_null()).map(|(id, v)| (*id, *v)).collect();
                base.sort();
                if prev.as_ref().is_some_and(|p| *p != base) {
                    ok = false;
                    break;
                }
                prev = Some(base);
                offs.push((store.id, store.index, store.x, off));
            }
            if !ok {
                continue;
            }
            offs.sort_by_key(|(_, _, _, off)| *off);
            if offs.iter().map(|(_, _, _, off)| *off).ne((0..(n as Dim)).map(|i| i * step)) {
                continue;
            }
            // Values stacked at row = off/step; the stack goes
            // after the last member so all values are declared.
            let mut ops = vec![OpId::NULL; n].into_boxed_slice();
            for (_, _, x, off) in &offs {
                ops[(off / step) as usize] = *x;
            }
            let ids: Set<OpId> = offs.iter().map(|(id, _, _, _)| *id).collect();
            let mut last_id = offs[0].0;
            let mut cur = offs[0].0;
            while !cur.is_null() {
                if ids.contains(&cur) {
                    last_id = cur;
                }
                cur = self.next_op(cur);
            }
            let vstore = self.insert_after(last_id, Op::Stack { ops });
            let mut prev = vstore;
            prev = self.insert_after(prev, Op::Const(Constant::new::<u16>(n.try_into().expect("vectorize: group overflows u16"))));
            let bytes = if elem.is_empty() {
                prev
            } else {
                let cn = prev;
                let mut dims: Vec<OpId> = Vec::with_capacity(elem.len() + 1);
                dims.push(cn);
                for &d in &elem {
                    let c = self.insert_after(
                        prev,
                        Op::Const(Constant::new::<u16>(d.try_into().expect("vectorize: width overflows u16"))),
                    );
                    dims.push(c);
                    prev = c;
                }
                self.insert_after(prev, Op::Stack { ops: dims.into_boxed_slice() })
            };
            let (_, base_index, _, _) = offs[0];
            self.insert_after(bytes, Op::Store { dst, dst_idx: base_index, src: vstore, bytes });
            for (id, _, _, _) in &offs {
                self.remove_op(*id);
            }
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

    /// Walk forward, find scalar ops over `Index` lanes of one vector that
    /// share the same op head, and bundle them: `Stack` the operands per
    /// position, rebuild the same op over the vectors, devectorize results.
    pub fn vectorize_ops_forward(&mut self, supported_lens: &[u8]) {
        // Process in descending length order for maximal vectorization first
        let mut supported: Vec<u8> = supported_lens.to_vec();
        supported.sort_unstable_by(|a, b| b.cmp(a));

        // Same op head (variant + payload, ignoring operands).
        let same_head = |a: &Op, b: &Op| match (a, b) {
            (Op::Unary { uop: x, .. }, Op::Unary { uop: y, .. }) => x == y,
            (Op::Cast { dtype: x, .. }, Op::Cast { dtype: y, .. }) => x == y,
            (Op::Bitcast { dtype: x, .. }, Op::Bitcast { dtype: y, .. }) => x == y,
            (Op::Binary { bop: x, .. }, Op::Binary { bop: y, .. }) => x == y,
            _ => false,
        };
        // Lane-operand check shared by every op kind.
        let lane_pat = Pat::index(Pat::bind('v'), 'k');

        // Group key: same source vec, same head, same lane position.
        // The template is the first member's op, reused for the rebuild.
        struct Group {
            source: OpId,
            template: Op,
            lane_pos: usize,
            members: Vec<OpId>,
        }
        let mut groups: Vec<Group> = Vec::new();

        // Fuel-capped: each iteration must vectorize one group; the cap
        // turns a no-progress bug into termination, never a hang.
        for _ in 0..10000 {
            groups.clear();

            // Full pass: collect ALL consumers into groups
            let mut op_id = self.head;
            while !op_id.is_null() {
                let is_candidate =
                    matches!(self.ops[op_id].op, Op::Unary { .. } | Op::Cast { .. } | Op::Bitcast { .. } | Op::Binary { .. });
                if is_candidate {
                    let operands: Vec<OpId> = self.ops[op_id].op.parameters().collect();
                    // First lane operand wins (x preferred for binary).
                    let mut found = None;
                    for (i, &o) in operands.iter().enumerate() {
                        if let Some(m) = self.match_pat(o, &lane_pat) {
                            found = Some((m.op('v'), i));
                            break;
                        }
                    }
                    if let Some((source, lane_pos)) = found {
                        if let Some(g) = groups
                            .iter_mut()
                            .find(|g| g.source == source && g.lane_pos == lane_pos && same_head(&g.template, &self.ops[op_id].op))
                        {
                            g.members.push(op_id);
                        } else {
                            groups.push(Group { source, template: self.ops[op_id].op.clone(), lane_pos, members: vec![op_id] });
                        }
                    }
                }
                op_id = self.next_op(op_id);
            }

            // Try groups in descending size order
            let mut indices: Vec<usize> = (0..groups.len()).filter(|&i| groups[i].members.len() >= 2).collect();
            indices.sort_by(|&a, &b| groups[b].members.len().cmp(&groups[a].members.len()));

            let mut applied = false;
            for &idx in &indices {
                let n = groups[idx].members.len();

                // Pick largest supported length <= n
                let Some(vec_len) = supported.iter().copied().find(|&l| (l as usize) <= n) else {
                    continue;
                };
                let vec_len = vec_len as usize;

                // Take the first vec_len entries
                let selected: Vec<OpId> = groups[idx].members.iter().take(vec_len).copied().collect();
                let lane_pos = groups[idx].lane_pos;
                let source_vec = groups[idx].source;
                let arity = groups[idx].template.parameters().count();

                let first = selected[0];

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
                    let lane_of = |consumer: OpId| self.ops[consumer].op.parameters().nth(lane_pos).expect("arity drift");
                    let late: Vec<OpId> =
                        selected.iter().map(|&c| lane_of(c)).filter(|&l| !self.op_is_before(l, first)).collect();
                    if !late.is_empty() {
                        let mut hoist_ok = self.op_is_before(source_vec, first);
                        for &l in &late {
                            if !hoist_ok {
                                break;
                            }
                            let mut cur = self.head;
                            while !cur.is_null() && cur != first {
                                if self.ops[cur].op.parameters().any(|p| p == l) {
                                    hoist_ok = false;
                                    break;
                                }
                                cur = self.next_op(cur);
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
                    for &c in &selected {
                        let operands: Vec<OpId> = self.ops[c].op.parameters().collect();
                        for (i, &o) in operands.iter().enumerate() {
                            if i != lane_pos && !self.op_is_before(o, first) {
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
                // the template op over the vectors.
                let mat: Vec<Vec<OpId>> = selected.iter().map(|&c| self.ops[c].op.parameters().collect()).collect();
                let mut vec_ops = Vec::with_capacity(arity);
                for i in 0..arity {
                    let stack: Box<[OpId]> = mat.iter().map(|row| row[i]).collect();
                    vec_ops.push(self.insert_before(first, Op::Stack { ops: stack }));
                }
                let mut rebuilt = groups[idx].template.clone();
                for (p, v) in rebuilt.parameters_mut().zip(vec_ops) {
                    *p = v;
                }
                let vec_op_id = self.insert_before(first, rebuilt);
                for (i, &consumer) in selected.iter().enumerate() {
                    self.ops[consumer].op = Op::Index { vec: vec_op_id, idx: i };
                }
                applied = true;
                break;
            }

            if !applied {
                break;
            }
        }
    }

    /// Walk backward, find Vectorize[X0..Xn] where all Xi are the same compute op.
    /// Replace Vectorize with compute_op(vectorize[inputs of Xi]),
    /// replace the first Xi with that vectorize of inputs, and keep walking.
    pub fn vectorize_ops_backward(&mut self, supported_lens: &[u8]) {
        let mut op_id = self.tail;
        while !op_id.is_null() {
            let ops = match &self.ops[op_id].op {
                Op::Stack { ops } => ops.clone(),
                _ => {
                    op_id = self.prev_op(op_id);
                    continue;
                }
            };

            let n = ops.len();
            if n < 2 || !supported_lens.contains(&(n as u8)) {
                op_id = self.prev_op(op_id);
                continue;
            }

            // Collect sources for each sub-op, then insert new vector ops
            // and remap consumers. Guard: no sub-op is also a source
            // (would create use-before-declaration in the IR).
            //
            // Insert new ops after the last operand in kernel order so all
            // operands are declared before the new Vectorize ops.
            match &self.ops[ops[0]].op {
                Op::Unary { uop, .. } => {
                    let uop = *uop;
                    let mut sources = Vec::with_capacity(n);
                    for &sub in ops.iter() {
                        match &self.ops[sub].op {
                            Op::Unary { x, uop: u } if *u == uop => sources.push(*x),
                            _ => {
                                sources.clear();
                                break;
                            }
                        }
                    }
                    if !sources.is_empty() && !sources.iter().any(|s| ops.contains(s)) {
                        let last = self.last_in_kernel_order(&sources);
                        let v_src = self.insert_after(last, Op::Stack { ops: sources.into_boxed_slice() });
                        let v_op = self.insert_after(v_src, Op::Unary { x: v_src, uop });
                        self.remap(op_id, v_op);
                    }
                }
                Op::Cast { dtype, .. } => {
                    let dtype = *dtype;
                    let mut sources = Vec::with_capacity(n);
                    for &sub in ops.iter() {
                        match &self.ops[sub].op {
                            Op::Cast { x, dtype: d } if *d == dtype => sources.push(*x),
                            _ => {
                                sources.clear();
                                break;
                            }
                        }
                    }
                    if !sources.is_empty() && !sources.iter().any(|s| ops.contains(s)) {
                        let last = self.last_in_kernel_order(&sources);
                        let v_src = self.insert_after(last, Op::Stack { ops: sources.into_boxed_slice() });
                        let v_op = self.insert_after(v_src, Op::Cast { x: v_src, dtype });
                        self.remap(op_id, v_op);
                    }
                }
                Op::Bitcast { dtype, .. } => {
                    let dtype = *dtype;
                    let mut sources = Vec::with_capacity(n);
                    for &sub in ops.iter() {
                        match &self.ops[sub].op {
                            Op::Bitcast { x, dtype: d } if *d == dtype => sources.push(*x),
                            _ => {
                                sources.clear();
                                break;
                            }
                        }
                    }
                    if !sources.is_empty() && !sources.iter().any(|s| ops.contains(s)) {
                        let last = self.last_in_kernel_order(&sources);
                        let v_src = self.insert_after(last, Op::Stack { ops: sources.into_boxed_slice() });
                        let v_op = self.insert_after(v_src, Op::Bitcast { x: v_src, dtype });
                        self.remap(op_id, v_op);
                    }
                }
                Op::Binary { bop, .. } => {
                    let bop = *bop;
                    let mut xs = Vec::with_capacity(n);
                    let mut ys = Vec::with_capacity(n);
                    for &sub in ops.iter() {
                        match &self.ops[sub].op {
                            Op::Binary { x, y, bop: b } if *b == bop => {
                                xs.push(*x);
                                ys.push(*y);
                            }
                            _ => {
                                xs.clear();
                                break;
                            }
                        }
                    }
                    if !xs.is_empty() && !xs.iter().any(|x| ops.contains(x)) && !ys.iter().any(|y| ops.contains(y)) {
                        let all: Vec<OpId> = xs.iter().chain(ys.iter()).copied().collect();
                        let last = self.last_in_kernel_order(&all);
                        let v_xs = self.insert_after(last, Op::Stack { ops: xs.into_boxed_slice() });
                        let v_ys = self.insert_after(v_xs, Op::Stack { ops: ys.into_boxed_slice() });
                        let v_op = self.insert_after(v_ys, Op::Binary { x: v_xs, y: v_ys, bop });
                        self.remap(op_id, v_op);
                    }
                }
                _ => {}
            }

            op_id = self.prev_op(op_id);
        }
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
