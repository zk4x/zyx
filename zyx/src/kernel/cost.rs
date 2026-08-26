// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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

use crate::{
    shape::Dim,
    DType, Map, Set,
    backend::DeviceInfo,
    kernel::{IDX_T, IdxKind, Kernel, MemLayout, MemScope, Op, OpId, ParamKind},
};
use nanoserde::{DeBin, SerBin};

/// The memory scope of a load source / store destination, which is either an
/// `Op::Storage` (kernel-internal buffer) or an `Op::Param` (kernel parameter:
/// a global or a variable).
fn mem_scope(op: &Op) -> MemScope {
    match op {
        Op::Storage { scope, .. } => *scope,
        Op::Param { kind: ParamKind::Variable, .. } => MemScope::Register,
        Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. } => MemScope::Global,
        _ => unreachable!("load/store operand must be a Storage or Param, got {op:?}"),
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Cost {
    /// Estimated execution time in microseconds.
    ///
    /// This is a learned cost estimate based on the kernel's
    /// characteristics (instruction count, memory access patterns,
    /// register usage, etc.). Lower values indicate better performance.
    pub(crate) cost: u64,
}

impl SerBin for Cost {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.cost.ser_bin(output);
    }
}

impl DeBin for Cost {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let cost = u64::de_bin(offset, bytes)?;
        Ok(Self { cost })
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Kernel {
    /// Get the estimated cost for this kernel.
    ///
    /// This method computes a cost estimate based on the kernel's
    /// characteristics:
    ///
    /// 1. First pass: compute reference counts and dtypes for register estimation
    /// 2. Second pass: instruction counting and register allocation simulation
    /// 3. Compute hardware-specific metrics (warp size, local memory, etc.)
    /// 4. Apply a learned cost model to predict execution time
    ///
    /// # Arguments
    ///
    /// * `dev_info` - Device information for hardware-specific parameters
    ///
    /// # Returns
    ///
    /// Returns a Cost estimate in microseconds.
    pub(crate) fn get_cost(&self, dev_info: &DeviceInfo) -> Cost {
        // First pass: compute reference counts and dtypes for register estimation
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::default();
        {
            let mut op_id = self.head;
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Asm { .. } => todo!(),
                    Op::Move { .. } | Op::Reduce { .. } => {
                        unreachable!()
                    }
                    Op::Stack { ref ops } => {
                        let dtype = dtypes[&ops[0]];
                        dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                    Op::Devectorize { vec, .. } => {
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
                    Op::Load { src, index, layout } => {
                        dtypes.insert(op_id, (dtypes[&src].0, layout));
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    Op::Store { dst, src: x, index, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(dst).or_insert(0) += 1;
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    Op::Cast { x, dtype } => {
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
                    Op::Mad { x, y, z } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                        *rcs.entry(z).or_insert(0) += 1;
                    }
                    Op::Index { .. } | Op::Loop { .. } => {
                        dtypes.insert(op_id, (DType::U32, MemLayout::Scalar));
                    }
                    Op::ReduceTile { x, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::MatmulTile { x, y } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                    }
                    Op::TransposeTile { x } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::If { condition } => {
                        *rcs.entry(condition).or_insert(0) += 1;
                    }
                    Op::Barrier | Op::EndIf | Op::EndLoop => {}
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
                Op::Asm { .. } => todo!(),
                Op::Storage { scope: MemScope::Register, .. } => true,
                Op::Load { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Mad { .. }
                | Op::Stack { .. }
                | Op::Wmma { .. }
                | Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::Loop { .. }
                | Op::Devectorize { .. }
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Const(_)
                | Op::Index { .. } => true,
                Op::Store { .. } | Op::EndLoop | Op::Barrier | Op::If { .. } | Op::EndIf => false,
                Op::Move { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
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
            if (matches!(op, Op::Index { .. } | Op::Loop { .. })
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
                Op::Cast { .. } | Op::Unary { .. } | Op::Binary { .. } => {
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                }
                Op::Mad { .. } => {
                    wi_ops += 2 * loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += 2 * loop_mult;
                    }
                }
                Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::EndIf
                | Op::Devectorize { .. }
                | Op::Stack { .. }
                | Op::Move { .. }
                | Op::Reduce { .. }
                | Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::Asm { .. }
                | Op::TransposeTile { .. } => {}
                Op::Load { src, index, layout } => {
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                    let scope = mem_scope(&self.ops[src].op);
                    let total_elements = loop_mult * layout.n_elements();
                    match scope {
                        MemScope::Global => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size()  as i64;
                            n_scoped_load_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { .. }) {
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
                            } else if let Op::Index { .. } = self.ops[index].op {
                                glb_load_lidx_stride_weighted += n_bits;
                                glb_load_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Local | MemScope::CircularReader | MemScope::CircularWriter => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size()  as i64;
                            n_scoped_load_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { .. }) {
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
                            } else if let Op::Index { .. } = self.ops[index].op {
                                loc_load_lidx_stride_weighted += n_bits;
                                loc_load_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Register => {
                            n_scoped_load_bits[2] += total_elements * dtypes[&op_id].0.bit_size()  as i64;
                        }
                    }
                }
                Op::Store { dst, index, layout, .. } => {
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    let scope = mem_scope(&self.ops[dst].op);
                    match scope {
                        MemScope::Global => {
                            let n_bits = loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size()  as i64;
                            n_scoped_store_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { .. }) {
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
                            } else if let Op::Index { .. } = self.ops[index].op {
                                glb_store_lidx_stride_weighted += n_bits;
                                glb_store_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Local | MemScope::CircularReader | MemScope::CircularWriter => {
                            let n_bits = loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size()  as i64;
                            n_scoped_store_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { .. }) {
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
                            } else if let Op::Index { .. } = self.ops[index].op {
                                loc_store_lidx_stride_weighted += n_bits;
                                loc_store_lidx_stride_weight += n_bits;
                            }
                        }
                        MemScope::Register => {
                            n_scoped_store_bits[2] += loop_mult * layout.n_elements() * dtypes[&op_id].0.bit_size()  as i64
                        }
                    }
                }
                Op::Index { axis, kind: scope } => match scope {
                    IdxKind::Group(len) => {
                        // Dynamic dims are `-1`; autotune substitutes 42 (see `alloc_buffers`).
                        gws[axis as usize] = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap_or(42)
                    }
                    IdxKind::Local(len) => lws[axis as usize] = len,
                    IdxKind::Warp(_) => todo!(),
                },
                Op::Loop { len: len_id } => {
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    if let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) {
                        loop_mult *= len;
                        latest_loop_lengths.push(len);
                    }
                    let depth = latest_loop_lengths.len()  as i64;
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
                Op::If { .. } => {
                    wi_branches += loop_mult;
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                }
            }

            // Track peak register bytes
            let bytes: u64 =
                reg_slots.iter().filter(|(r, _)| *r > 0).map(|(_, dt)| u64::from(dt.0.bit_size() / 8) * dt.1.n_elements() as u64).sum();
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
        let cost = Cost::predict_time_us(
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
        let cost = cost.max(1.0)  as i64;

        Cost { cost: cost as u64 }
    }
}
