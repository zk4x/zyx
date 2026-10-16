// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Pad index optimization.
//!
//! This module provides index padding optimizations for kernels,
//! which pad global indices to hardware-friendly sizes.
//!
//! Padding indices is useful for:
//!
//! - Enabling efficient tiling when tensor dimensions aren't multiples
//!   of the tile size
//! - Guarding out-of-range loads
//! - Skipping out-of-range stores with conditionals

use crate::{
    Set,
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, RangeKind},
    shape::Dim,
};

use super::autotune::Optimization;

/// Pad indices to hardware-friendly sizes (e.g., 32 for CUDA warps).
#[derive(Debug)]
pub struct PadIndex {
    /// Pairs of (index_op_id, target_size) for each padding.
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
        kernel.pad_range(idx_id, pad_to);
    }
}

impl Kernel {
    /// Pads a global index up to a multiple of `pad_to`, guarding out-of-range loads
    /// and skipping out-of-range stores.
    ///
    /// The pad amount is computed symbolically (`(pad_to - len % pad_to) % pad_to`
    /// as IR), so unresolvable lengths pad exactly with no `resolve_const`.
    /// This extends the range length so the grid covers full tiles. OOB reads
    /// are redirected to element 0 (safe). OOB stores are wrapped in a
    /// boolean-length `Op::Loop` (conditional) and skipped entirely.
    ///
    /// Useful for tiling: when a tensor dimension isn't a multiple of the tile size,
    /// pad the index so the grid covers full tiles, and OOB threads compute garbage
    /// but never write it to memory.
    ///
    /// # Panics
    /// - If `gidx_id` is not an `Op::Index` node.
    pub(crate) fn pad_range(&mut self, gidx_id: OpId, pad_to: Dim) {
        debug_assert!(pad_to > 0, "pad_range: pad_to {pad_to} <= 0");
        // 1. Extend the index length to the next multiple of pad_to.
        // The limit (original length) doubles as the OOB guard comparison.
        let Op::Range { axis, kind } = self.ops[gidx_id].op else {
            panic!("pad_index: op is not an Index");
        };
        let (limit, new_kind) = match kind {
            RangeKind::Group(len) => {
                let pad_to_c = self.insert_before(gidx_id, Op::Const(Constant::idx(pad_to)));
                let rem = self.insert_before(gidx_id, Op::Binary { x: len, y: pad_to_c, bop: BOp::Mod });
                let sub = self.insert_before(gidx_id, Op::Binary { x: pad_to_c, y: rem, bop: BOp::Sub });
                let pad = self.insert_before(gidx_id, Op::Binary { x: sub, y: pad_to_c, bop: BOp::Mod });
                let new_len = self.insert_before(gidx_id, Op::Binary { x: len, y: pad, bop: BOp::Add });
                (len, RangeKind::Group(new_len))
            }
            RangeKind::Local(len) => {
                let current_len = Dim::from(len);
                let pad_len = (pad_to - current_len % pad_to) % pad_to;
                if pad_len == 0 {
                    return;
                }
                let new_len = len + u32::try_from(pad_len).expect("pad_len too large for local index");
                let limit = self.insert_before(gidx_id, Op::Const(Constant::idx(current_len)));
                (limit, RangeKind::Local(new_len))
            }
            RangeKind::Warp(_) => {
                panic!("pad_range: cannot pad a warp view; pad its underlying local range instead")
            }
            RangeKind::Unroll(_) => {
                panic!("pad_range: cannot pad an unroll range; pad its underlying group range instead")
            }
        };
        self.ops[gidx_id].op = Op::Range { axis, kind: new_kind };

        // 2. Walk all ops to guard loads and stores depending on this index
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            // Redirect OOB stores to trash element at index `limit`
            if let Op::Store { dst, dst_idx: store_idx, src: x, bytes } = self.ops[op_id].op.clone()
                && self.depends_on(store_idx, gidx_id, &mut Set::default())
            {
                let buf_len: Option<Dim> = match &self.ops[dst].op {
                    Op::Param { .. } => Some(self.shape(dst).iter().product()),
                    _ => None,
                };
                if let Some(buf_len) = buf_len {
                    let clen = self.insert_before(op_id, Op::Const(Constant::idx(buf_len)));
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    let cast_cond = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let one = self.insert_before(op_id, Op::Const(Constant::idx(1)));
                    let not_cond = self.insert_before(op_id, Op::Binary { x: one, y: cast_cond, bop: BOp::Sub });
                    let idx_term = self.insert_before(op_id, Op::Binary { x: store_idx, y: cast_cond, bop: BOp::Mul });
                    let lim_term = self.insert_before(op_id, Op::Binary { x: clen, y: not_cond, bop: BOp::Mul });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: idx_term, y: lim_term, bop: BOp::Add });
                    self.ops[op_id].op = Op::Store { dst, dst_idx: safe_idx, src: x, bytes };
                }
            }

            // Redirect OOB copies to trash element at index `limit` (dst side, like stores)
            if let Op::Copy { src: copy_src, src_idx: copy_sidx, dst, dst_idx: store_idx, bytes } = self.ops[op_id].op.clone()
                && self.depends_on(store_idx, gidx_id, &mut Set::default())
            {
                let buf_len: Option<Dim> = match &self.ops[dst].op {
                    Op::Param { .. } => Some(self.shape(dst).iter().product()),
                    _ => None,
                };
                if let Some(buf_len) = buf_len {
                    let clen = self.insert_before(op_id, Op::Const(Constant::idx(buf_len)));
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    let cast_cond = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let one = self.insert_before(op_id, Op::Const(Constant::idx(1)));
                    let not_cond = self.insert_before(op_id, Op::Binary { x: one, y: cast_cond, bop: BOp::Sub });
                    let idx_term = self.insert_before(op_id, Op::Binary { x: store_idx, y: cast_cond, bop: BOp::Mul });
                    let lim_term = self.insert_before(op_id, Op::Binary { x: clen, y: not_cond, bop: BOp::Mul });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: idx_term, y: lim_term, bop: BOp::Add });
                    self.ops[op_id].op = Op::Copy { src: copy_src, src_idx: copy_sidx, dst, dst_idx: safe_idx, bytes };
                }
            }

            // Guard loads: redirect OOB reads to element 0 (safe)
            if let Op::Load { src, src_idx: load_idx, bytes } = self.ops[op_id].op.clone() {
                if !bytes.is_null() {
                    // non-scalar accesses stay scalar-guardable only
                    op_id = next;
                    continue;
                }
                if self.depends_on(load_idx, gidx_id, &mut Set::default()) {
                    let cond = self.insert_before(op_id, Op::Binary { x: gidx_id, y: limit, bop: BOp::Cmplt });
                    let cast_idx = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: load_idx, y: cast_idx, bop: BOp::Mul });
                    let safe_load = self.insert_before(op_id, Op::Load { src, src_idx: safe_idx, bytes });
                    self.remap(op_id, safe_load);
                    self.remove_op(op_id);
                }
            }

            op_id = next;
        }
    }

    #[allow(unused)]
    pub(crate) fn pad_loop(&mut self, loop_id: OpId, pad_len: Dim) {
        if pad_len == 0 {
            return;
        }

        // 1. Extend the loop length
        let Op::Loop { len } = &self.ops[loop_id].op else {
            panic!("pad_loop: op is not a Loop");
        };
        let current_len = self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim).unwrap();
        let new_len = self.insert_before(loop_id, Op::Const(Constant::idx(current_len + pad_len)));
        self.ops[loop_id].op = Op::Loop { len: new_len };

        // 2. Create limit constant for comparison
        let limit = self.insert_before(loop_id, Op::Const(Constant::idx(current_len)));

        // 3. Walk all ops to guard loads and stores depending on this loop
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            // Redirect OOB stores to trash element at index `buf_len`
            if let Op::Store { dst, dst_idx: store_idx, src: x, bytes } = self.ops[op_id].op.clone()
                && self.depends_on(store_idx, loop_id, &mut Set::default())
            {
                let buf_len: Option<Dim> = match self.ops[dst].op {
                    Op::Param { .. } => todo!(),
                    Op::Storage { len, .. } => Some(len),
                    _ => None,
                };
                if let Some(buf_len) = buf_len {
                    let clen = self.insert_before(op_id, Op::Const(Constant::idx(buf_len)));
                    let cond = self.insert_before(op_id, Op::Binary { x: loop_id, y: limit, bop: BOp::Cmplt });
                    let cast_cond = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let one = self.insert_before(op_id, Op::Const(Constant::idx(1)));
                    let not_cond = self.insert_before(op_id, Op::Binary { x: one, y: cast_cond, bop: BOp::Sub });
                    let idx_term = self.insert_before(op_id, Op::Binary { x: store_idx, y: cast_cond, bop: BOp::Mul });
                    let lim_term = self.insert_before(op_id, Op::Binary { x: clen, y: not_cond, bop: BOp::Mul });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: idx_term, y: lim_term, bop: BOp::Add });
                    self.ops[op_id].op = Op::Store { dst, dst_idx: safe_idx, src: x, bytes };
                }
            }

            // Redirect OOB copies to trash element at index `buf_len` (dst side, like stores)
            if let Op::Copy { src: copy_src, src_idx: copy_sidx, dst, dst_idx: store_idx, bytes } = self.ops[op_id].op.clone()
                && self.depends_on(store_idx, loop_id, &mut Set::default())
            {
                let buf_len: Option<Dim> = match self.ops[dst].op {
                    Op::Param { .. } => todo!(),
                    Op::Storage { len, .. } => Some(len),
                    _ => None,
                };
                if let Some(buf_len) = buf_len {
                    let clen = self.insert_before(op_id, Op::Const(Constant::idx(buf_len)));
                    let cond = self.insert_before(op_id, Op::Binary { x: loop_id, y: limit, bop: BOp::Cmplt });
                    let cast_cond = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let one = self.insert_before(op_id, Op::Const(Constant::idx(1)));
                    let not_cond = self.insert_before(op_id, Op::Binary { x: one, y: cast_cond, bop: BOp::Sub });
                    let idx_term = self.insert_before(op_id, Op::Binary { x: store_idx, y: cast_cond, bop: BOp::Mul });
                    let lim_term = self.insert_before(op_id, Op::Binary { x: clen, y: not_cond, bop: BOp::Mul });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: idx_term, y: lim_term, bop: BOp::Add });
                    self.ops[op_id].op = Op::Copy { src: copy_src, src_idx: copy_sidx, dst, dst_idx: safe_idx, bytes };
                }
            }

            // Guard loads: redirect OOB reads to element 0 (safe)
            if let Op::Load { src, src_idx: load_idx, bytes } = self.ops[op_id].op.clone() {
                if !bytes.is_null() {
                    // non-scalar accesses stay scalar-guardable only
                    op_id = next;
                    continue;
                }
                if self.depends_on(load_idx, loop_id, &mut Set::default()) {
                    let cond = self.insert_before(op_id, Op::Binary { x: loop_id, y: limit, bop: BOp::Cmplt });
                    let cast_idx = self.insert_before(op_id, Op::Cast { x: cond, dtype: IDX_T });
                    let safe_idx = self.insert_before(op_id, Op::Binary { x: load_idx, y: cast_idx, bop: BOp::Mul });
                    let safe_load = self.insert_before(op_id, Op::Load { src, src_idx: safe_idx, bytes });
                    self.remap(op_id, safe_load);
                    self.remove_op(op_id);
                }
            }

            op_id = next;
        }
    }

    /// Make the `PadIndex` optimization: scan the kernel for indices whose
    /// lengths are not provably multiples of hardware-friendly sizes.
    pub fn opt_pad_index(&self, _dev_info: &DeviceInfo) -> Box<dyn Optimization> {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Range { kind, .. } = self.ops[op_id].op {
                match kind {
                    // Symbolic lengths skip only when provably a multiple.
                    RangeKind::Group(len) => {
                        for pad_to in [8, 16, 32] {
                            if !self.is_sym_multiple(len, pad_to) {
                                factors.push((op_id, pad_to));
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
                    // Unroll widths are exact lane counts — skip them entirely.
                    RangeKind::Unroll(_) => {}
                }
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
