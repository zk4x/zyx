// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Pad axis optimization.
//!
//! This module provides axis padding for kernels, which pads range and
//! loop lengths up to a multiple of a hardware-friendly size.
//!
//! Padding is useful for:
//!
//! - Enabling efficient tiling when tensor dimensions aren't multiples
//!   of the tile size
//! - Guarding out-of-range loads
//! - Skipping out-of-range stores with conditionals
//!
//! `pad_to(axis, multiple)` pads the length up to the next multiple
//! (`multiple = 32` pads up to a multiple of 32) via the round-up idiom
//! `((len + (multiple - 1)) / multiple) * multiple` as IR. Ranges pad on
//! every device; loops pad only on Tenstorrent (tile loops must run
//! full-tile trip counts). Lengths already provably a multiple (concrete
//! or symbolic, via `is_sym_multiple`) are left untouched; the trailing
//! multiply keeps padded lengths provably multiple, so no axis pads
//! twice. No `resolve_const` on this path.

use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

use super::autotune::Optimization;

/// Pad axes to hardware-friendly multiples (e.g., 32 for CUDA warps).
#[derive(Debug)]
pub struct PadIndex {
    /// Pairs of (axis_op_id, multiple) for each padding.
    pub factors: Vec<(OpId, Dim)>,
}

impl Optimization for PadIndex {
    fn nconfigs(&self) -> u64 {
        self.factors.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        if self.factors.is_empty() {
            return;
        }
        let (idx_id, pad_to) = self.factors[config as usize];
        kernel.pad_to(idx_id, pad_to);
    }
}

impl Kernel {
    /// Pads a range length up to a multiple of `multiple`, guarding
    /// out-of-range loads and stores. Ranges pad on every device; loops
    /// pad only on Tenstorrent (tile loops must run full-tile trip
    /// counts, other devices never extend loop bounds).
    ///
    /// The length is rounded up via `((len + (multiple - 1)) / multiple) *
    /// multiple` as IR. This extends the length so the grid covers full
    /// tiles. OOB reads are redirected to element 0 (safe). OOB stores
    /// are redirected to a trash element past the buffer end.
    ///
    /// Useful for tiling: when a tensor dimension isn't a multiple of the
    /// tile size, pad the axis so the grid covers full tiles, and OOB
    /// threads compute garbage but never write it to memory.
    ///
    /// # Panics
    /// - If `axis` is not a range (non-TT) or neither a range nor a loop (TT).
    /// - If `axis` is a warp view or unroll range (pad the underlying
    ///   local/group range instead).
    pub(crate) fn pad_to(&mut self, axis: OpId, multiple: Dim) {
        debug_assert!(multiple > 0, "pad_to: multiple {multiple} <= 0");
        match self.ops[axis].op {
            Op::Range { axis: range_axis, kind } => {
                // 1. Extend the index length to the next multiple.
                // The limit (original length) doubles as the OOB guard
                // comparison.
                let (limit, new_kind) = match kind {
                    RangeKind::Group(len) => {
                        if self.dtype(len) == crate::DType::Bool {
                            // Boolean lengths are 0/1 flags, never pad candidates.
                            return;
                        }
                        if self.is_sym_multiple(len, multiple) {
                            return;
                        }
                        // Index arithmetic is IDX_T throughout; a symbolic
                        // length of any int dtype is cast once, concrete
                        // lengths become IDX_T consts. The round-up below
                        // and the limit guard then always typecheck.
                        let len_idx = match self.resolve_const(len).and_then(Constant::as_dim) {
                            Some(l) => self.insert_before(axis, Op::Const(Constant::idx(l))),
                            None => self.insert_before(axis, Op::Cast { x: len, dtype: IDX_T }),
                        };
                        // Round up to the next multiple:
                        // ((len + (multiple - 1)) / multiple) * multiple.
                        // The trailing multiply by Const(multiple) makes
                        // the result provably a multiple
                        // (is_sym_multiple), so the scan never re-pads
                        // this axis.
                        let multiple_c = self.insert_before(axis, Op::Const(Constant::idx(multiple)));
                        let m1 = self.insert_before(axis, Op::Const(Constant::idx(multiple - 1)));
                        let up = self.insert_before(axis, Op::Binary { x: len_idx, y: m1, bop: BOp::Add });
                        let q = self.insert_before(axis, Op::Binary { x: up, y: multiple_c, bop: BOp::Div });
                        let new_len = self.insert_before(axis, Op::Binary { x: q, y: multiple_c, bop: BOp::Mul });
                        (len_idx, RangeKind::Group(new_len))
                    }
                    RangeKind::Local(len) => {
                        let current_len = Dim::from(len);
                        let pad_len = (multiple - current_len % multiple) % multiple;
                        if pad_len == 0 {
                            return;
                        }
                        let new_len = len + u32::try_from(pad_len).expect("pad_len too large for local index");
                        let limit = self.insert_before(axis, Op::Const(Constant::idx(current_len)));
                        (limit, RangeKind::Local(new_len))
                    }
                    RangeKind::Warp(_) => {
                        panic!("pad_to: cannot pad a warp view; pad its underlying local range instead")
                    }
                };
                self.ops[axis].op = Op::Range { axis: range_axis, kind: new_kind };

                // TT pads lengths only: physical tile buffers absorb halo
                // by construction, and guard loops would make split refuse
                // the axis (see tt_tile). Other backends execute seed
                // indices literally, so guards stay load-bearing there.
                #[cfg(feature = "tenstorrent")]
                let tt = matches!(self.dev, crate::backend::Dev::TT(_));
                #[cfg(not(feature = "tenstorrent"))]
                let tt = false;
                if tt {
                    return;
                }

                // 2. Walk all ops after this axis and guard the loads and
                // stores depending on it. Stores are wrapped unconditionally:
                // the guard condition tests the thread position, never the
                // (rewritable) index, so wrapping is exact per thread and
                // later index rewrites cannot detach it.
                let mut op_id = self.next_op(axis);
                while !op_id.is_null() {
                    let next = self.next_op(op_id);

                    // Skip OOB stores entirely: wrap the store in a 0-or-1-trip
                    // loop over the in-bounds condition. No single trash
                    // index serves every destination buffer (multi-dim
                    // outputs share no common past-the-end element, and
                    // buffer lengths are unrealized here), so OOB stores
                    // simply never execute. Register destinations are
                    // thread-private and never OOB — leave them alone so
                    // accumulator patterns stay intact for later passes.
                    if let Op::Store { dst, .. } = self.ops[op_id].op.clone()
                        && !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. })
                    {
                        let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                        self.insert_before(op_id, Op::Loop { len: cond });
                        self.insert_after(op_id, Op::EndLoop);
                    }

                    // Skip OOB copies entirely (dst side, like stores).
                    if let Op::Copy { dst, .. } = self.ops[op_id].op.clone()
                        && !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. })
                    {
                        let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                        self.insert_before(op_id, Op::Loop { len: cond });
                        self.insert_after(op_id, Op::EndLoop);
                    }

                    // Guard loads: redirect OOB reads to element 0 (safe).
                    // A Select on the index keeps the guard a straight-line
                    // value op (no conditional scope for later passes to
                    // reason about); OOB threads load element 0 and their
                    // garbage never reaches memory (stores stay guarded).
                    if let Op::Load { src, src_idx: load_idx } = self.ops[op_id].op.clone() {
                        if !self.shape(load_idx).is_empty() {
                            // non-scalar accesses stay scalar-guardable only
                            op_id = next;
                            continue;
                        }
                        if self.depends_on(load_idx, axis, &mut Set::default()) {
                            let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                            let zero = self.insert_before(op_id, Op::Const(Constant::idx(0)));
                            let safe_idx = self.insert_before(op_id, Op::Select { cond, if_true: load_idx, if_false: zero });
                            let safe_load = self.insert_before(op_id, Op::Load { src, src_idx: safe_idx });
                            self.remap(op_id, safe_load);
                            self.remove_op(op_id);
                        }
                    }

                    op_id = next;
                }
            }
            Op::Loop { .. } => {
                // Loop padding is Tenstorrent-only: tile loops must run
                // full-tile trip counts. Other devices never extend loop
                // bounds. TT also needs element-level OOB checks: its
                // 32x32 tiles span memory, so edge tiles read OOB elements
                // even for in-bounds tile iterations (still open).
                #[cfg(feature = "tenstorrent")]
                let tt = matches!(self.dev, crate::backend::Dev::TT(_));
                #[cfg(not(feature = "tenstorrent"))]
                let tt = false;
                if !tt {
                    return;
                }
                // 1. Extend the loop length to the next multiple.
                let Op::Loop { len } = self.ops[axis].op else {
                    panic!("pad_to: op is not a loop");
                };
                if self.dtype(len) == crate::DType::Bool {
                    // Boolean lengths are 0/1 flags, never pad candidates.
                    return;
                }
                if self.is_sym_multiple(len, multiple) {
                    return;
                }
                // Round up to the next multiple (see the range arm: the
                // trailing multiply keeps the result provably a multiple).
                // Lengths are cast to IDX_T once, like group ranges.
                let len_idx = match self.resolve_const(len).and_then(Constant::as_dim) {
                    Some(l) => self.insert_before(axis, Op::Const(Constant::idx(l))),
                    None => self.insert_before(axis, Op::Cast { x: len, dtype: IDX_T }),
                };
                let multiple_c = self.insert_before(axis, Op::Const(Constant::idx(multiple)));
                let m1 = self.insert_before(axis, Op::Const(Constant::idx(multiple - 1)));
                let up = self.insert_before(axis, Op::Binary { x: len_idx, y: m1, bop: BOp::Add });
                let q = self.insert_before(axis, Op::Binary { x: up, y: multiple_c, bop: BOp::Div });
                let new_len = self.insert_before(axis, Op::Binary { x: q, y: multiple_c, bop: BOp::Mul });
                self.ops[axis].op = Op::Loop { len: new_len };
                // TT pads lengths only (see the range arm): guard loops
                // would make split refuse the axis.
                if tt {
                    return;
                }

                // 2. The original length doubles as the OOB guard comparison.
                let limit = len_idx;

                // 3. Walk the loop body and guard the loads and stores
                // depending on it (see the range arm: store wrapping
                // is unconditional so later index rewrites cannot detach it).
                // Only the body: the trip count extension affects
                // per-iteration stores, while stores after the loop never
                // execute per-iteration. The guard references the loop
                // variable, which leaves scope at the loop's own EndLoop,
                // so wrapping past it is use-before-declaration.
                let mut op_id = self.next_op(axis);
                let mut depth = 1u32;
                while !op_id.is_null() {
                    let next = self.next_op(op_id);
                    match self.ops[op_id].op {
                        Op::Loop { .. } => depth += 1,
                        Op::EndLoop => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }

                    // Skip OOB stores entirely (see the range arm; register
                    // destinations are thread-private, leave them alone).
                    if let Op::Store { dst, .. } = self.ops[op_id].op.clone()
                        && !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. })
                    {
                        let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                        self.insert_before(op_id, Op::Loop { len: cond });
                        self.insert_after(op_id, Op::EndLoop);
                    }

                    // Skip OOB copies entirely (dst side, like stores).
                    if let Op::Copy { dst, .. } = self.ops[op_id].op.clone()
                        && !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. })
                    {
                        let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                        self.insert_before(op_id, Op::Loop { len: cond });
                        self.insert_after(op_id, Op::EndLoop);
                    }

                    // Guard loads: redirect OOB reads to element 0 (safe,
                    // Select-on-index like the range arm: no conditional
                    // scope, OOB garbage never reaches guarded stores).
                    if let Op::Load { src, src_idx: load_idx } = self.ops[op_id].op.clone() {
                        if !self.shape(load_idx).is_empty() {
                            // non-scalar accesses stay scalar-guardable only
                            op_id = next;
                            continue;
                        }
                        if self.depends_on(load_idx, axis, &mut Set::default()) {
                            let cond = self.insert_before(op_id, Op::Binary { x: axis, y: limit, bop: BOp::Cmplt });
                            let zero = self.insert_before(op_id, Op::Const(Constant::idx(0)));
                            let safe_idx = self.insert_before(op_id, Op::Select { cond, if_true: load_idx, if_false: zero });
                            let safe_load = self.insert_before(op_id, Op::Load { src, src_idx: safe_idx });
                            self.remap(op_id, safe_load);
                            self.remove_op(op_id);
                        }
                    }

                    op_id = next;
                }
            }
            _ => panic!("pad_to: op is not a range"),
        }
    }

    /// Make the `PadIndex` optimization: scan the kernel for ranges whose
    /// lengths are not provably multiples of hardware-friendly sizes.
    /// Loops are scanned only on Tenstorrent (tile loops must run
    /// full-tile trip counts); elsewhere loop bounds are never extended.
    pub fn opt_pad(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "tenstorrent")]
        let tt = matches!(self.dev, crate::backend::Dev::TT(_));
        #[cfg(not(feature = "tenstorrent"))]
        let tt = false;
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Range { kind, .. } => match kind {
                    // Symbolic lengths skip only when provably a multiple.
                    // Boolean lengths are 0/1 flags, never pad candidates.
                    RangeKind::Group(len) => {
                        if self.dtype(len) != crate::DType::Bool {
                            for pad_to in [8, 16, 32] {
                                if !self.is_sym_multiple(len, pad_to) {
                                    factors.push((op_id, pad_to));
                                }
                            }
                        }
                    }
                    RangeKind::Local(len) => {
                        let len = Dim::from(len);
                        for pad_to in [8, 16, 32] {
                            if len % pad_to != 0 {
                                factors.push((op_id, pad_to));
                            }
                        }
                    }
                    // Warp views can't be padded — skip them entirely.
                    RangeKind::Warp(_) => {}
                },
                Op::Loop { len, .. } => {
                    if tt && self.dtype(len) != crate::DType::Bool {
                        for pad_to in [8, 16, 32] {
                            if !self.is_sym_multiple(len, pad_to) {
                                factors.push((op_id, pad_to));
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = next;
        }
        Box::new(PadIndex { factors })
    }

    pub(crate) fn depends_on(&self, expr: OpId, target: OpId, visited: &mut Set<OpId>) -> bool {
        if expr == target || !visited.insert(expr) {
            return expr == target;
        }
        match self.at(expr) {
            Op::Const(_) | Op::Range { .. } | Op::Storage { .. } | Op::Loop { .. } | Op::EndLoop => false,
            op => op.parameters().any(|p| self.depends_on(p, target, visited)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::kernel::{Dev, MemScope};

    /// Loop padding extends the trip count, so only the loop body may
    /// reference the loop variable in a guard. Stores after the loop
    /// never execute per-iteration and must stay guard-free: a guard
    /// over the closed loop variable fails `verify` (use outside its
    /// scope reads as use-before-declaration).
    #[test]
    fn pad_loop_guard_stays_in_body() {
        let mut k = Kernel::new(Dev::Auto);
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let init = k.const_val(0.0f32);
        k.store(acc, init, 0i64);
        k.loop_over(4i64, |k, lv| {
            let x = k.load(inp, lv);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let result = k.load(acc, 0i64);
        k.store(out, result, 0i64);

        let loop_id = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Loop { .. }).then_some(id))
            .expect("loop present");
        k.pad_to(loop_id, 8);

        // Every use of the loop variable must sit inside its body.
        let mut stack: Vec<OpId> = Vec::new();
        let mut id = k.head;
        while !id.is_null() {
            if matches!(k.at(id), Op::Loop { .. }) {
                stack.push(id);
            }
            if id != loop_id && !stack.contains(&loop_id) && k.at(id).parameters().any(|p| p == loop_id) {
                panic!("use of loop {loop_id:?} outside its body at {id:?}: {:?}", k.at(id));
            }
            if matches!(k.at(id), Op::EndLoop) {
                stack.pop();
            }
            id = k.next_op(id);
        }
        k.verify();
    }
}
