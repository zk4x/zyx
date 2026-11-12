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
    kernel::{IDX_T, Kernel, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind},
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
    pub fn features(&self) -> Features {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("cost features");
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

        // Dtypes, shapes and reference counts: the canonical single pass.
        // `Self::shape` walks an op's entire dependency cone, so calling it
        // per op here (as this pass used to) made the feature pass quadratic
        // in the IR size and duplicated `compute_dtypes_and_rcs`.
        let (dtypes, rcs) = self.compute_dtypes_and_rcs();

        // Second pass: instruction counting + register allocation simulation
        let mut wi_compute_ops = 0;
        let mut wi_ops = 0;
        let mut n_scoped_load_bits = [0i64; 3];
        let mut n_scoped_store_bits = [0i64; 3];
        let mut wi_barriers = 0i64;
        let mut gws = [1i64; 3];
        let mut lws = [1u32; 3];
        let mut loop_mult = 1i64;
        // Unroll lanes in an op's cone: one vector op is one scalar op per lane
        // after devectorization, so cost scales with this.
        let mut lane_lens: Map<OpId, i64> = Map::default();
        let mut latest_loop_lengths: Vec<Dim> = Vec::new();
        let mut max_loop_depth = 0i64;

        let mut reg_slots: Vec<(u32, (DType, Vec<Dim>))> = Vec::new(); // (rc, dtype)
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

        // Concrete element count of an access shape. Every dim here
        // resolves from const-built bytes descriptors; a negative dim
        // is malformed IR — fail loudly.
        let concrete_len = |shape: &[Dim]| -> Dim {
            shape
                .iter()
                .map(|&d| {
                    assert!(d >= 0, "cost: symbolic access width");
                    d
                })
                .product()
        };
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
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Select { .. }
                | Op::Stack { .. }
                | Op::Wmma { .. }
                | Op::Index { .. }
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Const(_)
                | Op::Unroll { .. }
                | Op::Range { .. } => true,
                Op::Store { .. } | Op::Copy { .. } | Op::TT(_) | Op::EndLoop | Op::Barrier => false,
                // Counted loops produce the induction value; boolean
                // loops (conditionals) produce nothing, like `If` did.
                Op::Loop { len } => self.dtype(len) != DType::Bool,
                Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. } => todo!(),
                Op::Reduce { .. } => true,
                Op::Source(_) | Op::GPU(_) | Op::Spirv(_) => todo!(),
                Op::PTX(ref p) => match p.as_ref() {
                    PTXOp::Bytes(_) => todo!(),
                    PTXOp::VLdMatrix { .. } => true,
                    // Effect only.
                    PTXOp::WarpSync => false,
                    // A register move, no memory.
                    PTXOp::Shfl { .. } => false,
                    // The cost model runs before the device lowering: these
                    // are never priced.
                    PTXOp::LdMatrix { .. }
                    | PTXOp::CvtaShared { .. }
                    | PTXOp::SharedBase { .. }
                    | PTXOp::LdShared { .. }
                    | PTXOp::StShared { .. } => todo!("cost: device-lowered PTX is never priced"),
                },
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
                // Graph-only marker, erased at kernelization: never present.
                Op::Detach { .. } => {
                    todo!()
                }
            };
            if produces && let Some(&rc) = rcs.get(&op_id) {
                let Some(dtype) = dtypes.get(&op_id).cloned() else {
                    panic!("cost: no dtype for producing op {op_id}: {:?}", self.ops[op_id].op);
                };
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

            // Lane multiplicity: a vector op expands into one scalar op per
            // lane when the device epilogue unfolds the unroll lanes, so its
            // cost is the vector cost times the product of the lane lengths in
            // its operand cone. Applied by scaling `loop_mult` around the
            // accounting below and dividing it back out, which keeps the
            // loop-nesting updates in that block untouched.
            let lane_mult: i64 = op
                .parameters()
                .filter(|p| !p.is_null())
                .map(|p| lane_lens.get(&p).copied().unwrap_or(1))
                .fold(1, |a, b| a.saturating_mul(b));
            let own_lanes = match *op {
                Op::Unroll { len, .. } => i64::from(len),
                _ => lane_mult,
            };
            lane_lens.insert(op_id, own_lanes.max(1));
            // The lane factors compound along an op chain (each op's cone is
            // its operands' cones combined), so a long elementwise chain
            // overflows `i64` in debug builds. The product is a cost
            // multiplier, not an allocation, so saturate instead of wrapping:
            // any kernel this large ranks last, which is the point.
            let lane_mult = lane_mult.clamp(1, 1 << 32);
            loop_mult = loop_mult.saturating_mul(lane_mult).min(1 << 40);

            // Instruction counting
            match self.ops[op_id].op {
                Op::Cast { .. } | Op::Bitcast { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Select { .. } => {
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                }
                // The reduce expands into one add per lane pair: the chain
                // cost is (lane_len - 1) additions, lane_len the product
                // of the reduce's unroll lens.
                Op::Reduce { reduce_axis, .. } => {
                    let mut lane_len: u64 = 1;
                    let mut stack = vec![reduce_axis];
                    let mut seen: Set<OpId> = Set::default();
                    while let Some(p) = stack.pop() {
                        if !seen.insert(p) {
                            continue;
                        }
                        match self.at(p) {
                            Op::Unroll { len, .. } => lane_len *= u64::from(*len),
                            Op::Binary { x, y, .. } => {
                                stack.push(*x);
                                stack.push(*y);
                            }
                            // Legacy pre-linearize axes carry no lane space.
                            Op::Const(_) => {}
                            _ => {}
                        }
                    }
                    let adds = (lane_len.saturating_sub(1)) as i64;
                    wi_ops += loop_mult * adds;
                    wi_compute_ops += loop_mult * adds;
                }
                Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Index { .. }
                | Op::Stack { .. }
                | Op::Unroll { .. }
                | Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::TT { .. }
                | Op::Asm { .. } => {}
                Op::Load { src, src_idx: index, .. } => {
                    let access_shape = dtypes[&op_id].1.clone();
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                    let scope = mem_scope(&self.ops[src].op);
                    // Vector width already lives in the access shape, so the
                    // lane factor must not be applied to traffic as well.
                    let total_elements = loop_mult / lane_mult * concrete_len(&access_shape);
                    match scope {
                        MemScope::Global => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_load_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index).unwrap_or_default();
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
                            let strides = self.get_strides(index).unwrap_or_default();
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
                Op::Store { dst, dst_idx: index, .. } => {
                    let access_shape = dtypes[&op_id].1.clone();
                    let n_elements = concrete_len(&access_shape);
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    let scope = mem_scope(&self.ops[dst].op);
                    // Vector width already lives in the access shape.
                    let traffic_mult = loop_mult / lane_mult;
                    match scope {
                        MemScope::Global => {
                            let n_bits = traffic_mult * n_elements * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_store_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index).unwrap_or_default();
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
                            let n_bits = traffic_mult * n_elements * dtypes[&op_id].0.bit_size() as i64;
                            n_scoped_store_bits[1] += n_bits;
                            let strides = self.get_strides(index).unwrap_or_default();
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
                            n_scoped_store_bits[2] += traffic_mult * n_elements * dtypes[&op_id].0.bit_size() as i64
                        }
                    }
                }
                Op::Copy { src, dst, .. } => {
                    // A copy reads once through src and writes once through
                    // dst: account read bits to the src scope, write bits to
                    // the dst scope, at single-transfer weight. No stride
                    // heuristics (those tune DRAM coalescing, not
                    // storage-to-storage moves).
                    wi_ops += loop_mult * 2;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 2;
                    }
                    let n_elements = concrete_len(&dtypes[&op_id].1);
                    for (side, is_read) in [(src, true), (dst, false)] {
                        let bits = loop_mult / lane_mult * n_elements * dtypes[&side].0.bit_size() as i64;
                        let slot = match mem_scope(&self.ops[side].op) {
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
                Op::Source(_) | Op::GPU(_) | Op::Spirv(_) => todo!(),
                Op::PTX(ref p) => match p.as_ref() {
                    PTXOp::Bytes(_) => todo!(),
                    // One shared-memory instruction per fragment bundle.
                    PTXOp::VLdMatrix { .. } => {
                        wi_ops += loop_mult;
                        wi_compute_ops += loop_mult;
                    }
                    // A warp-level barrier: one instruction, and a stall.
                    PTXOp::WarpSync => {
                        wi_ops += loop_mult;
                        wi_barriers += loop_mult;
                    }
                    // One shuffle instruction.
                    PTXOp::Shfl { .. } => {
                        wi_ops += loop_mult;
                        wi_compute_ops += loop_mult;
                    }
                    PTXOp::LdMatrix { .. }
                    | PTXOp::CvtaShared { .. }
                    | PTXOp::SharedBase { .. }
                    | PTXOp::LdShared { .. }
                    | PTXOp::StShared { .. } => todo!("cost: device-lowered PTX is never priced"),
                },
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
                // Graph-only marker, erased at kernelization: never present.
                Op::Detach { .. } => {
                    todo!()
                }
            }

            // Undo the lane scaling: the loop nesting factor must persist, the
            // lane factor must not.
            loop_mult /= lane_mult;

            // Track peak register bytes
            let bytes: u64 = reg_slots
                .iter()
                .filter(|(r, _)| *r > 0)
                .map(|(_, dt)| u64::from(dt.0.bit_size() / 8) * concrete_len(&dt.1) as u64)
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

        // How much `devectorize` will expand the body: the product of every
        // unroll lane length in the kernel.
        let lane_product: u64 = self
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Unroll { len, .. } => Some(u64::from(*len)),
                _ => None,
            })
            .product();
        let lane_product = lane_product.min(u64::from(u32::MAX)) as u32;

        // The raw feature vector: one definition shared by the search, the
        // benchmark collector and the trainer.
        Features {
            num_groups: num_groups as u32,
            wi_per_group,
            wi_ops: wi_ops as u32,
            wi_compute_ops: wi_compute_ops as u32,
            wi_barriers: wi_barriers as u32,
            wi_global_load_bits: wi_global_load_bits as u32,
            wi_global_store_bits: wi_global_store_bits as u32,
            wi_local_load_bits: wi_local_load_bits as u32,
            wi_local_store_bits: wi_local_store_bits as u32,
            wi_peak_reg_bytes: wi_peak_reg_bytes as u32,
            wi_branches: wi_branches as u32,
            wi_global_load_lidx_stride: glb_load_lidx_stride as u32,
            wi_global_store_lidx_stride: glb_store_lidx_stride as u32,
            wi_local_load_lidx_stride: loc_load_lidx_stride as u32,
            wi_local_store_lidx_stride: loc_store_lidx_stride as u32,
            warp_size: dev_info.warp_size as u32,
            max_local_threads: dev_info.max_local_threads,
            max_register_bytes: dev_info.max_register_bytes as u32,
            wi_register_load_bits: wi_register_load_bits as u32,
            wi_register_store_bits: wi_register_store_bits as u32,
            gws0: gws[0] as u32,
            gws1: gws[1] as u32,
            gws2: gws[2] as u32,
            lws0: lws[0],
            lws1: lws[1],
            lws2: lws[2],
            max_loop_depth: max_loop_depth as u32,
            preferred_vector_size: dev_info.preferred_vector_size as u32,
            local_mem_size: dev_info.local_mem_size as u32,
            lane_product,
        }
    }

    /// Learned cost of this kernel in nanoseconds, floored at 1 ns so that only
    /// a genuinely free kernel can tie. The floor used to be one *microsecond*,
    /// which collapsed every fast candidate onto the same integer and left the
    /// launch order to insertion order.
    pub fn base_cost(&self) -> u64 {
        (self.features().predict_us() * 1000.0).max(1.0) as u64
    }
}

/// The raw feature vector the learned cost model consumes.
///
/// One definition for every consumer: [`Kernel::features`] fills it, the
/// search ranks with [`Features::predict_us`], and the benchmark collector
/// writes it out under [`Features::NAMES`] so the trainer's columns and the
/// collector's keys can never drift from the model's inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    /// Number of workgroups in the launch: the product of the `Group` range
    /// lengths.
    pub num_groups: u32,
    /// Work-items per workgroup: the product of the `Local` range lengths.
    pub wi_per_group: u32,
    /// Instructions executed per work-item, after loop unrolling and lane
    /// expansion.
    pub wi_ops: u32,
    /// The subset of [`Self::wi_ops`] that is compute rather than index
    /// arithmetic.
    pub wi_compute_ops: u32,
    /// Barriers executed per work-item.
    pub wi_barriers: u32,
    /// Global-memory load traffic per work-item, in bits.
    pub wi_global_load_bits: u32,
    /// Global-memory store traffic per work-item, in bits.
    pub wi_global_store_bits: u32,
    /// Local/circular-memory load traffic per work-item, in bits.
    pub wi_local_load_bits: u32,
    /// Local/circular-memory store traffic per work-item, in bits.
    pub wi_local_store_bits: u32,
    /// Peak live register bytes in the simulated register allocation.
    pub wi_peak_reg_bytes: u32,
    /// Conditional branches per work-item (boolean `Loop`s).
    pub wi_branches: u32,
    /// Traffic-weighted average linear-index stride of global loads.
    pub wi_global_load_lidx_stride: u32,
    /// Traffic-weighted average linear-index stride of global stores.
    pub wi_global_store_lidx_stride: u32,
    /// Traffic-weighted average linear-index stride of local loads.
    pub wi_local_load_lidx_stride: u32,
    /// Traffic-weighted average linear-index stride of local stores.
    pub wi_local_store_lidx_stride: u32,
    /// Device warp/wavefront width, in work-items.
    pub warp_size: u32,
    /// Device maximum number of work-items per workgroup.
    pub max_local_threads: u32,
    /// Device private-memory (register) budget per work-item, in bytes.
    pub max_register_bytes: u32,
    /// Register/private-memory load traffic per work-item, in bits.
    pub wi_register_load_bits: u32,
    /// Register/private-memory store traffic per work-item, in bits.
    pub wi_register_store_bits: u32,
    /// Global work size on axis 0.
    pub gws0: u32,
    /// Global work size on axis 1.
    pub gws1: u32,
    /// Global work size on axis 2.
    pub gws2: u32,
    /// Local work size on axis 0.
    pub lws0: u32,
    /// Local work size on axis 1.
    pub lws1: u32,
    /// Local work size on axis 2.
    pub lws2: u32,
    /// Deepest counted-loop nesting in the kernel.
    pub max_loop_depth: u32,
    /// Device-preferred vector width, in bytes.
    pub preferred_vector_size: u32,
    /// Device local/shared memory size, in bytes.
    pub local_mem_size: u32,
    /// Product of every `Op::Unroll` length in the kernel.
    ///
    /// `devectorize` expands the body by exactly this factor, so it is the
    /// gap between the kernel the search ranks (pre-lowering) and the kernel
    /// that runs. A candidate with 128 lanes and 61 ops lowers to ~7.8k scalar
    /// ops; without this feature the model cannot see that cost.
    pub lane_product: u32,
}

impl Features {
    /// Field names, in [`Features::values`] order. The collector's record keys
    /// and the trainer's columns are both derived from this list.
    pub const NAMES: [&'static str; 30] = [
        "num_groups",
        "wi_per_group",
        "wi_ops",
        "wi_compute_ops",
        "wi_barriers",
        "wi_global_load_bits",
        "wi_global_store_bits",
        "wi_local_load_bits",
        "wi_local_store_bits",
        "wi_peak_reg_bytes",
        "wi_branches",
        "wi_global_load_lidx_stride",
        "wi_global_store_lidx_stride",
        "wi_local_load_lidx_stride",
        "wi_local_store_lidx_stride",
        "warp_size",
        "max_local_threads",
        "max_register_bytes",
        "wi_register_load_bits",
        "wi_register_store_bits",
        "gws0",
        "gws1",
        "gws2",
        "lws0",
        "lws1",
        "lws2",
        "max_loop_depth",
        "preferred_vector_size",
        "local_mem_size",
        "lane_product",
    ];

    /// The raw values, in model order.
    pub fn values(&self) -> [u32; 30] {
        [
            self.num_groups,
            self.wi_per_group,
            self.wi_ops,
            self.wi_compute_ops,
            self.wi_barriers,
            self.wi_global_load_bits,
            self.wi_global_store_bits,
            self.wi_local_load_bits,
            self.wi_local_store_bits,
            self.wi_peak_reg_bytes,
            self.wi_branches,
            self.wi_global_load_lidx_stride,
            self.wi_global_store_lidx_stride,
            self.wi_local_load_lidx_stride,
            self.wi_local_store_lidx_stride,
            self.warp_size,
            self.max_local_threads,
            self.max_register_bytes,
            self.wi_register_load_bits,
            self.wi_register_store_bits,
            self.gws0,
            self.gws1,
            self.gws2,
            self.lws0,
            self.lws1,
            self.lws2,
            self.max_loop_depth,
            self.preferred_vector_size,
            self.local_mem_size,
            self.lane_product,
        ]
    }

    /// Predicted execution time in microseconds.
    pub fn predict_us(&self) -> f64 {
        let [
            num_groups,
            wi_per_group,
            wi_ops,
            wi_compute_ops,
            wi_barriers,
            wi_global_load_bits,
            wi_global_store_bits,
            wi_local_load_bits,
            wi_local_store_bits,
            wi_peak_reg_bytes,
            wi_branches,
            wi_global_load_lidx_stride,
            wi_global_store_lidx_stride,
            wi_local_load_lidx_stride,
            wi_local_store_lidx_stride,
            warp_size,
            max_local_threads,
            max_register_bytes,
            wi_register_load_bits,
            wi_register_store_bits,
            gws0,
            gws1,
            gws2,
            lws0,
            lws1,
            lws2,
            max_loop_depth,
            preferred_vector_size,
            local_mem_size,
            lane_product,
        ] = self.values();
        predict_time_us(
            num_groups,
            wi_per_group,
            wi_ops,
            wi_compute_ops,
            wi_barriers,
            wi_global_load_bits,
            wi_global_store_bits,
            wi_local_load_bits,
            wi_local_store_bits,
            wi_peak_reg_bytes,
            wi_branches,
            wi_global_load_lidx_stride,
            wi_global_store_lidx_stride,
            wi_local_load_lidx_stride,
            wi_local_store_lidx_stride,
            warp_size,
            max_local_threads,
            max_register_bytes,
            wi_register_load_bits,
            wi_register_store_bits,
            gws0,
            gws1,
            gws2,
            lws0,
            lws1,
            lws2,
            max_loop_depth,
            preferred_vector_size,
            local_mem_size,
            lane_product,
        )
    }
}
