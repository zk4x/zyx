// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Cost estimation for kernel autotuning.
//!
//! This module provides cost estimation utilities for evaluating kernel
//! performance during autotuning. The cost model considers:
//!
//! - Instruction count
//! - Compute operations
//! - Memory access patterns (global/local/register)
//! - Register allocation
//! - Loop depth and parallelism
//! - Hardware-specific parameters (warp size, local memory, etc.)
//!
//! The cost model is learned from actual kernel execution times and
//! used to guide the autotuning search.

use super::predict_cost::predict_time_us;
use crate::{
    DType, Map, Set,
    kernel::{IDX_T, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TTOp},
    shape::Dim,
};

impl Kernel {
    /// Predict the execution time of this kernel in microseconds.
    ///
    /// Complete package: walks the kernel IR twice (reference counts + dtypes,
    /// then instruction counting and register-allocation simulation) and feeds
    /// the extracted features to the learned cost model in `predict_cost.rs`.
    /// Hardware parameters come from `dev_info`. Lower values indicate better
    /// performance.
    ///
    /// Variable-backed (dynamic) group lengths cost as the 42 placeholder: the
    /// real dims only exist at launch and differ across launches, so the
    /// placeholder is a fixed convention the trained model is calibrated
    /// against (TVM-style: only concrete shapes are ever costed).
    pub fn base_cost(&self) -> u64 {
        // The memory scope of a load source / store destination, which is
        // either an `Op::Storage` (kernel-internal buffer) or an `Op::Param`
        // (kernel parameter: a global or a variable).
        fn mem_scope(op: &Op) -> MemScope {
            match op {
                Op::Storage { scope, .. } => *scope,
                Op::Param { kind: ParamKind::Variable, .. } => MemScope::Register,
                Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. } => MemScope::Global,
                _ => unreachable!("load/store operand must be a Storage or Param, got {op:?}"),
            }
        }

        let dev_info = self.dev_info();

        // First pass: compute reference counts and dtypes for register estimation
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::default();
        {
            let mut op_id = self.head;
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                        // Same rule as compute_dtypes_and_rcs: result takes
                        // ops[0]'s dtype/layout, ops are consumed operands.
                        // Operand-free Asm/LLK is an effect-only call: no
                        // value, no entry.
                        if !ops.is_empty() {
                            let (dtype, layout) = dtypes[&ops[0]];
                            dtypes.insert(op_id, (dtype, layout));
                            for &x in ops.iter() {
                                *rcs.entry(x).or_insert(0) += 1;
                            }
                        }
                    }
                    Op::TT(TTOp::LLKReduce { cb_in, cb_sc, slot, x, scaler, .. }) => {
                        // Lowered reduce: same rule over cb_in.
                        let (dtype, layout) = dtypes[&cb_in];
                        dtypes.insert(op_id, (dtype, layout));
                        for &f in &[cb_in, cb_sc, slot, x, scaler] {
                            *rcs.entry(f).or_insert(0) += 1;
                        }
                    }
                    Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                        // Lowered fused broadcast: same rule over cb_a.
                        let (dtype, layout) = dtypes[&cb_a];
                        dtypes.insert(op_id, (dtype, layout));
                        for &f in &[cb_a, cb_b, mx, plain] {
                            *rcs.entry(f).or_insert(0) += 1;
                        }
                    }
                    Op::Expand { .. }
                    | Op::Permute { .. }
                    | Op::Flip { .. }
                    | Op::Narrow { .. }
                    | Op::Reshape { .. }
                    | Op::Pad { .. }
                    | Op::Reduce { .. } => {
                        unreachable!()
                    }
                    Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                        unreachable!()
                    }
                    Op::Stack { ref ops } => {
                        let dtype = dtypes[&ops[0]];
                        dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                    Op::Index { vec, .. } => {
                        let dtype = dtypes[&vec];
                        dtypes.insert(op_id, (dtype.0, MemLayout::Scalar));
                        *rcs.entry(vec).or_insert(0) += 1;
                    }
                    Op::Const(x) => {
                        dtypes.insert(op_id, (x.dtype(), MemLayout::Scalar));
                    }
                    Op::Param { dtype, .. } => {
                        dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                    }
                    Op::Storage { dtype, .. } => {
                        dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                    }
                    Op::Wmma { c, a, b, .. } => {
                        dtypes.insert(op_id, (DType::F32, MemLayout::Vector(4)));
                        *rcs.entry(a).or_insert(0) += 1;
                        *rcs.entry(b).or_insert(0) += 1;
                        *rcs.entry(c).or_insert(0) += 1;
                    }
                    Op::Load { src } => {
                        dtypes.insert(op_id, dtypes[&src]);
                    }
                    Op::Store { dst, src: x } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(dst).or_insert(0) += 1;
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Copy { src, dst } => {
                        dtypes.insert(op_id, dtypes[&src]);
                        *rcs.entry(src).or_insert(0) += 1;
                        *rcs.entry(dst).or_insert(0) += 1;
                    }
                    Op::GEP { x, index, layout } => {
                        dtypes.insert(op_id, (dtypes[&x].0, layout));
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    Op::Cast { x, dtype } => {
                        dtypes.insert(op_id, (dtype, dtypes[&x].1));
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Bitcast { x, dtype } => {
                        dtypes.insert(op_id, (dtype, dtypes[&x].1));
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Unary { x, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Binary { x, y, bop } => {
                        let dtype = if bop.returns_bool() {
                            (DType::Bool, dtypes[&x].1)
                        } else {
                            dtypes[&x]
                        };
                        dtypes.insert(op_id, dtype);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                    }
                    Op::Range { .. } => {
                        dtypes.insert(op_id, (DType::U32, MemLayout::Scalar));
                    }
                    Op::Loop { len } => {
                        dtypes.insert(op_id, (DType::U32, MemLayout::Scalar));
                        // Boolean length = the old `If` condition: live
                        // across the body, so it keeps its refcount.
                        // Counted-loop lengths were never refcounted here.
                        if self.dtype(len) == DType::Bool {
                            *rcs.entry(len).or_insert(0) += 1;
                        }
                    }
                    Op::TT(ref ttop) => match ttop {
                        TTOp::ReduceTile { x, scaler, acc, .. } => {
                            dtypes.insert(op_id, dtypes[acc]);
                            *rcs.entry(*x).or_insert(0) += 1;
                            *rcs.entry(*scaler).or_insert(0) += 1;
                            *rcs.entry(*acc).or_insert(0) += 1;
                        }
                        TTOp::MatmulTile { x, y, acc } => {
                            dtypes.insert(op_id, dtypes[acc]);
                            *rcs.entry(*x).or_insert(0) += 1;
                            *rcs.entry(*y).or_insert(0) += 1;
                            *rcs.entry(*acc).or_insert(0) += 1;
                        }
                        TTOp::TransposeTile { x } => {
                            dtypes.insert(op_id, dtypes[x]);
                            *rcs.entry(*x).or_insert(0) += 1;
                        }
                        TTOp::BroadcastTile { x, .. } => {
                            dtypes.insert(op_id, dtypes[x]);
                            *rcs.entry(*x).or_insert(0) += 1;
                        }
                        TTOp::ReserveBack { cb, .. }
                        | TTOp::PushBack { cb, .. }
                        | TTOp::WaitFront { cb, .. }
                        | TTOp::PopFront { cb, .. } => {
                            dtypes.insert(op_id, dtypes[cb]);
                            *rcs.entry(*cb).or_insert(0) += 1;
                        }
                        TTOp::LLK { ops, .. } => {
                            // Opaque effect call: no value, but the CB
                            // operands stay live (same rule as the Asm
                            // arm above).
                            if !ops.is_empty() {
                                dtypes.insert(op_id, dtypes[&ops[0]]);
                                for &x in ops.iter() {
                                    *rcs.entry(x).or_insert(0) += 1;
                                }
                            }
                        }
                        TTOp::LLKReduce { cb_in, cb_sc, slot, x, scaler, .. } => {
                            dtypes.insert(op_id, dtypes[cb_in]);
                            for f in [cb_in, cb_sc, slot, x, scaler] {
                                *rcs.entry(*f).or_insert(0) += 1;
                            }
                        }
                        TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. } => {
                            dtypes.insert(op_id, dtypes[cb_a]);
                            for f in [cb_a, cb_b, mx, plain] {
                                *rcs.entry(*f).or_insert(0) += 1;
                            }
                        }
                        TTOp::MathLock
                        | TTOp::MathUnlock
                        | TTOp::PackLock
                        | TTOp::PackUnlock
                        | TTOp::NocReadBarrier
                        | TTOp::NocWriteBarrier
                        | TTOp::ReduceUninit
                        | TTOp::EndReader
                        | TTOp::EndCompute => {}
                    },
                    Op::Source(_) | Op::GPU(_) => todo!(),
                    Op::Barrier | Op::EndLoop => {}
                }
                op_id = self.next_op(op_id);
            }
        }

        // Second pass: instruction counting + register allocation simulation
        let mut wi_compute_ops = 0;
        let mut wi_ops = 0;
        let mut n_scoped_load_bits = [0i64; 3];
        let mut n_scoped_store_bits = [0i64; 3];
        let mut wi_barriers = 0i64;
        let mut gws = [1i64; 3];
        let mut lws = [1u32; 3];
        let mut loop_mult = 1i64;
        let mut latest_loop_lengths: Vec<Dim> = Vec::new();
        let mut max_loop_depth = 0i64;

        let mut reg_slots: Vec<(u32, (DType, MemLayout))> = Vec::new(); // (rc, dtype)
        let mut reg_map: Map<OpId, usize> = Map::default();
        let mut indexing_ops: Set<OpId> = Set::default();
        let mut wi_peak_reg_bytes = 0u64;
        let mut wi_branches = 0i64;
        let mut glb_load_lidx_stride_weighted = 0i64;
        let mut glb_load_lidx_stride_weight = 0i64;
        let mut glb_store_lidx_stride_weighted = 0i64;
        let mut glb_store_lidx_stride_weight = 0i64;
        let mut loc_load_lidx_stride_weighted = 0i64;
        let mut loc_load_lidx_stride_weight = 0i64;
        let mut loc_store_lidx_stride_weighted = 0i64;
        let mut loc_store_lidx_stride_weight = 0i64;

        let mut op_id = self.head;
        while !op_id.is_null() {
            // Register allocation: allocate if this op produces a value
            let produces = match self.ops[op_id].op {
                // Value-producing Asm takes ops[0]'s dtype (see the
                // first pass above); operand-free Asm is an
                // effect-only call and produces no register.
                Op::Asm { ref ops, .. } => !ops.is_empty(),
                Op::Storage { scope: MemScope::Register, .. } => true,
                Op::Load { .. }
                | Op::GEP { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Stack { .. }
                | Op::Wmma { .. }
                | Op::TT(TTOp::ReduceTile { .. })
                | Op::TT(TTOp::MatmulTile { .. })
                | Op::TT(TTOp::TransposeTile { .. })
                | Op::TT(TTOp::BroadcastTile { .. })
                | Op::Index { .. }
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Const(_)
                | Op::Range { .. } => true,
                Op::Store { .. }
                | Op::Copy { .. }
                | Op::TT(TTOp::ReserveBack { .. })
                | Op::TT(TTOp::PushBack { .. })
                | Op::TT(TTOp::WaitFront { .. })
                | Op::TT(TTOp::PopFront { .. })
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::LLKReduce { .. })
                | Op::TT(TTOp::LLKBcast { .. })
                | Op::EndLoop
                | Op::Barrier => false,
                // Counted loops produce the induction value; boolean
                // loops (conditionals) produce nothing, like `If` did.
                Op::Loop { len } => self.dtype(len) != DType::Bool,
                Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
                Op::Source(_) | Op::GPU(_) => todo!(),
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
            };
            if produces && let Some(&rc) = rcs.get(&op_id) {
                let dtype = dtypes[&op_id];
                let idx = reg_slots.iter().position(|(r, dt)| *r == 0 && *dt == dtype).unwrap_or_else(|| {
                    let i = reg_slots.len();
                    reg_slots.push((0, dtype));
                    i
                });
                reg_slots[idx].0 = rc;
                reg_map.insert(op_id, idx);
            }

            // Decrement RC for each operand
            for param in self.ops[op_id].op.parameters() {
                if let Some(&p) = reg_map.get(&param)
                    && reg_slots[p].0 > 0
                {
                    reg_slots[p].0 -= 1;
                }
            }

            let op = &self.ops[op_id].op;

            // Is this indexing or compute?
            // (Boolean loops are conditionals, never indexing — like `If`.)
            if (matches!(op, Op::Range { .. })
                || matches!(op, Op::Loop { len } if self.dtype(*len) != DType::Bool)
                || (op.parameters().count() > 0 && op.parameters().all(|p| indexing_ops.contains(&p))))
            {
                indexing_ops.insert(op_id);
            }
            if let Op::Const(c) = op
                && c.dtype() == IDX_T
            {
                indexing_ops.insert(op_id);
            }

            // Instruction counting
            match self.ops[op_id].op {
                Op::Cast { .. } | Op::Bitcast { .. } | Op::Unary { .. } | Op::Binary { .. } => {
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                }
                Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::GEP { .. }
                | Op::Index { .. }
                | Op::Stack { .. }
                | Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::Reduce { .. }
                | Op::TT { .. }
                | Op::Asm { .. } => {}
                Op::Load { src } => {
                    // The GEP carries the location triple; the rest of this
                    // arm is unchanged from the inline-triple form.
                    let (buf, index, layout) = match self.ops[src].op {
                        Op::GEP { x, index, layout } => (x, index, layout),
                        _ => todo!("cost: Load src must be a GEP"),
                    };
                    let src = buf;
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                    let scope = mem_scope(&self.ops[src].op);
                    let total_elements = loop_mult * layout.n_elements();
                    match scope {
                        MemScope::Global => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_load_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Range { .. }) {
                                        Some(*st)
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    strides.iter().find_map(|(oid, (_, st))| {
                                        if oid.is_null() || *st == 0 {
                                            return None;
                                        }
                                        if matches!(self.ops[*oid].op, Op::Loop { .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
                                });
                            if let Some(st) = stride {
                                glb_load_lidx_stride_weighted += st * n_bits;
                                glb_load_lidx_stride_weight += n_bits;
                            } else if let Op::Range { .. } = self.ops[index].op {
                                glb_load_lidx_stride_weighted += n_bits;
                                glb_load_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Local | MemScope::Circular => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_load_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Range { .. }) {
                                        Some(*st)
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    strides.iter().find_map(|(oid, (_, st))| {
                                        if oid.is_null() || *st == 0 {
                                            return None;
                                        }
                                        if matches!(self.ops[*oid].op, Op::Loop { .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
                                });
                            if let Some(st) = stride {
                                loc_load_lidx_stride_weighted += st * n_bits;
                                loc_load_lidx_stride_weight += n_bits;
                            } else if let Op::Range { .. } = self.ops[index].op {
                                loc_load_lidx_stride_weighted += n_bits;
                                loc_load_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Register => {
                            n_scoped_load_bits[2] += total_elements * dtypes[&op_id].0.bit_size() as i64;
                        }
                    }
                }
                Op::Store { dst, .. } => {
                    // Pre-linearize whole-view stores name the Param
                    // directly (Scalar layout, NULL index — the old inline
                    // form exactly); post-linearize the GEP carries the triple.
                    let (buf, index, layout) = match self.ops[dst].op {
                        Op::GEP { x, index, layout } => (x, index, layout),
                        Op::Param { .. } => (dst, OpId::NULL, MemLayout::Scalar),
                        _ => todo!("cost: Store dst must be a GEP or Param"),
                    };
                    let dst = buf;
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    let scope = mem_scope(&self.ops[dst].op);
                    match scope {
                        MemScope::Global => {
                            let n_bits = loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_store_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Range { .. }) {
                                        Some(*st)
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    strides.iter().find_map(|(oid, (_, st))| {
                                        if oid.is_null() || *st == 0 {
                                            return None;
                                        }
                                        if matches!(self.ops[*oid].op, Op::Loop { .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
                                });
                            if let Some(st) = stride {
                                glb_store_lidx_stride_weighted += st * n_bits;
                                glb_store_lidx_stride_weight += n_bits;
                            } else if let Op::Range { .. } = self.ops[index].op {
                                glb_store_lidx_stride_weighted += n_bits;
                                glb_store_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Local | MemScope::Circular => {
                            let n_bits = loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_store_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Range { .. }) {
                                        Some(*st)
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| {
                                    strides.iter().find_map(|(oid, (_, st))| {
                                        if oid.is_null() || *st == 0 {
                                            return None;
                                        }
                                        if matches!(self.ops[*oid].op, Op::Loop { .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
                                });
                            if let Some(st) = stride {
                                loc_store_lidx_stride_weighted += st * n_bits;
                                loc_store_lidx_stride_weight += n_bits;
                            } else if let Op::Range { .. } = self.ops[index].op {
                                loc_store_lidx_stride_weighted += n_bits;
                                loc_store_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Register => {
                            n_scoped_store_bits[2] += loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size() as i64
                        }
                    }
                }
                Op::Copy { src, dst } => {
                    // A copy reads once through src and writes once through
                    // dst: account read bits to the src scope, write bits to
                    // the dst scope, at single-transfer weight. No stride
                    // heuristics (those tune DRAM coalescing, not
                    // storage-to-storage moves).
                    wi_ops += loop_mult * 2;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 2;
                    }
                    for (gep, is_read) in [(src, true), (dst, false)] {
                        let (buf, layout) = match self.ops[gep].op {
                            Op::GEP { x, layout, .. } => (x, layout),
                            _ => todo!("cost: Copy side must be a GEP"),
                        };
                        let bits = loop_mult * layout.n_elements() * dtypes[&gep].0.bit_size() as i64;
                        let slot = match mem_scope(&self.ops[buf].op) {
                            MemScope::Global => 0,
                            MemScope::Local | MemScope::Circular => 1,
                            MemScope::Register => 2,
                        };
                        if is_read {
                            n_scoped_load_bits[slot] += bits;
                        } else {
                            n_scoped_store_bits[slot] += bits;
                        }
                    }
                }
                Op::Range { axis, kind: scope } => match scope {
                    RangeKind::Group(len) => {
                        // Dynamic dims are `-1`; autotune substitutes 42 (see `alloc_buffers`).
                        gws[axis as usize] = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap_or(42)
                    }
                    RangeKind::Local(len) => lws[axis as usize] = len,
                    // A warp is a view over a local range — adds no threads.
                    RangeKind::Warp(_) => {}
                },
                Op::Loop { len: len_id } => {
                    // Boolean length = the old `If`: branch cost, neutral
                    // stack entry (a const bool would resolve to 0/1 via
                    // `as_dim` and corrupt `loop_mult`, so check dtype first).
                    if self.dtype(len_id) == DType::Bool {
                        wi_branches += loop_mult;
                        wi_ops += loop_mult * 3;
                        if !indexing_ops.contains(&op_id) {
                            wi_compute_ops += loop_mult * 3;
                        }
                        latest_loop_lengths.push(1);
                    } else {
                        wi_ops += loop_mult * 3;
                        if !indexing_ops.contains(&op_id) {
                            wi_compute_ops += loop_mult * 3;
                        }
                        if let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) {
                            loop_mult *= len;
                            latest_loop_lengths.push(len);
                        } else {
                            // Dynamic dim length at cost time (e.g. a variable
                            // loop bound; autotune substitutes the real value at
                            // launch). Keep the loop stack balanced with a
                            // neutral entry.
                            latest_loop_lengths.push(1);
                        }
                    }
                    let depth = latest_loop_lengths.len() as i64;
                    if depth > max_loop_depth {
                        max_loop_depth = depth;
                    }
                }
                Op::EndLoop => {
                    loop_mult /= latest_loop_lengths.pop().unwrap();
                }
                Op::Wmma { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    let warp = u64::from(dev_info.warp_size);
                    let cost = (m * n * k) / warp;
                    wi_ops += loop_mult * cost as Dim;
                    if !indexing_ops.contains(&op_id) {
                        // TODO multiply by some constant
                        wi_compute_ops += loop_mult * cost as Dim;
                    }
                }
                Op::Barrier => {
                    wi_barriers += loop_mult;
                }
                Op::Source(_) | Op::GPU(_) => todo!(),
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
            }

            // Track peak register bytes
            let bytes: u64 = reg_slots
                .iter()
                .filter(|(r, _)| *r > 0)
                .map(|(_, dt)| u64::from(dt.0.bit_size() / 8) * dt.1.n_elements() as u64)
                .sum();
            if bytes > wi_peak_reg_bytes {
                wi_peak_reg_bytes = bytes;
            }

            op_id = self.next_op(op_id);
        }

        let wi_global_load_bits = n_scoped_load_bits[0];
        let wi_local_load_bits = n_scoped_load_bits[1];
        let wi_register_load_bits = n_scoped_load_bits[2];
        let wi_global_store_bits = n_scoped_store_bits[0];
        let wi_local_store_bits = n_scoped_store_bits[1];
        let wi_register_store_bits = n_scoped_store_bits[2];

        let num_groups: Dim = gws.iter().product();
        let wi_per_group: u32 = lws.iter().product();

        let glb_load_lidx_stride = if glb_load_lidx_stride_weight > 0 {
            glb_load_lidx_stride_weighted as f64 / glb_load_lidx_stride_weight as f64
        } else {
            0.0
        };
        let glb_store_lidx_stride = if glb_store_lidx_stride_weight > 0 {
            glb_store_lidx_stride_weighted as f64 / glb_store_lidx_stride_weight as f64
        } else {
            0.0
        };

        let loc_load_lidx_stride = if loc_load_lidx_stride_weight > 0 {
            loc_load_lidx_stride_weighted as f64 / loc_load_lidx_stride_weight as f64
        } else {
            0.0
        };
        let loc_store_lidx_stride = if loc_store_lidx_stride_weight > 0 {
            loc_store_lidx_stride_weighted as f64 / loc_store_lidx_stride_weight as f64
        } else {
            0.0
        };

        // Learned cost model: rank 0..1 within variant * 1_000_000 (2000 DT leaves + Ridge)
        let cost = predict_time_us(
            num_groups as u32,
            wi_per_group as u32,
            wi_ops as u32,
            wi_compute_ops as u32,
            wi_barriers as u32,
            wi_global_load_bits as u32,
            wi_global_store_bits as u32,
            wi_local_load_bits as u32,
            wi_local_store_bits as u32,
            wi_peak_reg_bytes as u32,
            wi_branches as u32,
            glb_load_lidx_stride as u32,
            glb_store_lidx_stride as u32,
            loc_load_lidx_stride as u32,
            loc_store_lidx_stride as u32,
            dev_info.warp_size as u32,
            dev_info.max_local_threads as u32,
            dev_info.max_register_bytes as u32,
            wi_register_load_bits as u32,
            wi_register_store_bits as u32,
            gws[0] as u32,
            gws[1] as u32,
            gws[2] as u32,
            lws[0] as u32,
            lws[1] as u32,
            lws[2] as u32,
            max_loop_depth as u32,
            dev_info.preferred_vector_size as u32,
            dev_info.local_mem_size as u32,
        );
        cost.max(1.0) as u64
    }
}
