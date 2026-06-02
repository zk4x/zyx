// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map, Set,
    backend::DeviceInfo,
    kernel::{IDX_T, Kernel, Op, OpId, Scope},
};
use nanoserde::{DeBin, SerBin};

/*#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, DeBin, SerBin)]
pub struct Cost {
    pub cost: u64,
}*/

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, DeBin, SerBin)]
pub struct Cost {
    pub cost: u64,
    num_groups: u64,
    wi_per_group: u64,
    wi_ops: u64,
    wi_compute_ops: u64,
    wi_barriers: u64,
    wi_global_load_bits: u64,
    wi_global_store_bits: u64,
    wi_local_load_bits: u64,
    wi_local_store_bits: u64,
    wi_peak_reg_bytes: u64,
    wi_branches: u64,
    wi_global_load_lidx_stride: u64,
    wi_global_store_lidx_stride: u64,
    wi_local_load_lidx_stride: u64,
    wi_local_store_lidx_stride: u64,

    warp_size: u64,
    max_local_threads: u64,
    max_register_bytes: u64,
}

impl Cost {
    pub fn debug(&self) {
        println!(
            "cost={}, num_groups={}, wi_per_group={}, wi_ops={}, wi_compute_ops={}, wi_barriers={}, wi_global_load_bits={}, wi_global_store_bits={}
wi_local_load_bits={}, wi_local_store_bits={}, wi_peak_reg_bytes={}, wi_branches={}, wi_global_load_lidx_stride={}, wi_global_store_lidx_stride={}, wi_local_load_lidx_stride={}, wi_local_store_lidx_stride={}, warp_size={}, max_local_threads={}, max_register_bytes={}",
            self.cost,
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
            self.max_register_bytes
        );
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
    pub fn get_cost(&self, dev_info: &DeviceInfo) -> Cost {
        // First pass: compute reference counts and dtypes for register estimation
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut dtypes: Map<OpId, (DType, u16)> = Map::default();
        {
            let mut op_id = self.head;
            while !op_id.is_null() {
                let op = self.at(op_id);
                match op {
                    Op::Devectorize { .. }
                    | Op::ConstView { .. }
                    | Op::StoreView { .. }
                    | Op::LoadView { .. }
                    | Op::Move { .. }
                    | Op::Reduce { .. } => unreachable!(),
                    Op::Vectorize { ops } => {
                        let dtype = dtypes[&ops[0]];
                        dtypes.insert(op_id, (dtype.0, ops.len() as u16));
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                    Op::Const(x) => {
                        dtypes.insert(op_id, (x.dtype(), 1));
                    }
                    Op::Define { dtype, .. } => {
                        dtypes.insert(op_id, (*dtype, 1));
                    }
                    &Op::Wmma { c, a, b, .. } => {
                        dtypes.insert(op_id, (DType::F32, 4));
                        *rcs.entry(a).or_insert(0) += 1;
                        *rcs.entry(b).or_insert(0) += 1;
                        *rcs.entry(c).or_insert(0) += 1;
                    }
                    &Op::Load { src, index, vlen } => {
                        dtypes.insert(op_id, (dtypes[&src].0, vlen));
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    &Op::Store { dst, x, index, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(dst).or_insert(0) += 1;
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    &Op::Cast { x, dtype } => {
                        dtypes.insert(op_id, (dtype, dtypes[&x].1));
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    &Op::Unary { x, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    &Op::Binary { x, y, bop } => {
                        let dtype = if bop.returns_bool() {
                            (DType::Bool, dtypes[&x].1)
                        } else {
                            dtypes[&x]
                        };
                        dtypes.insert(op_id, dtype);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                    }
                    &Op::Mad { x, y, z } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                        *rcs.entry(z).or_insert(0) += 1;
                    }
                    Op::Index { .. } | Op::Loop { .. } => {
                        dtypes.insert(op_id, (DType::U32, 1));
                    }
                    &Op::If { condition } => {
                        *rcs.entry(condition).or_insert(0) += 1;
                    }
                    Op::Barrier { .. } | Op::EndIf | Op::EndLoop => {}
                }
                op_id = self.next_op(op_id);
            }
        }

        // Second pass: instruction counting + register allocation simulation
        let mut wi_compute_ops = 0;
        let mut wi_ops = 0;
        let mut n_scoped_load_bits = [0u64; 3];
        let mut n_scoped_store_bits = [0u64; 3];
        let mut wi_barriers = 0u64;
        let mut gws = [1u64; 3];
        let mut lws = [1u64; 3];
        let mut loop_mult = 1u64;
        let mut latest_loop_lengths: Vec<u64> = Vec::new();

        let mut reg_slots: Vec<(u32, (DType, u16))> = Vec::new(); // (rc, dtype)
        let mut reg_map: Map<OpId, usize> = Map::default();
        let mut indexing_ops: Set<OpId> = Set::default();
        let mut wi_peak_reg_bytes = 0u64;
        let mut wi_branches = 0u64;
        let mut glb_load_lidx_stride_weighted = 0u64;
        let mut glb_load_lidx_stride_weight = 0u64;
        let mut glb_store_lidx_stride_weighted = 0u64;
        let mut glb_store_lidx_stride_weight = 0u64;
        let mut loc_load_lidx_stride_weighted = 0u64;
        let mut loc_load_lidx_stride_weight = 0u64;
        let mut loc_store_lidx_stride_weighted = 0u64;
        let mut loc_store_lidx_stride_weight = 0u64;

        let mut op_id = self.head;
        while !op_id.is_null() {
            let op = self.at(op_id);

            // Register allocation: allocate if this op produces a value
            let produces = match op {
                Op::Define { scope, .. } if *scope == Scope::Register => true,
                Op::Load { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Mad { .. }
                | Op::Vectorize { .. }
                | Op::Wmma { .. }
                | Op::Loop { .. }
                | Op::Devectorize { .. }
                | Op::Define { .. }
                | Op::Const(_)
                | Op::Index { .. } => true,
                Op::Store { .. } | Op::EndLoop | Op::Barrier { .. } | Op::If { .. } | Op::EndIf => false,
                Op::ConstView(_) => todo!(),
                Op::LoadView(_) => todo!(),
                Op::StoreView { .. } => todo!(),
                Op::Move { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
            };
            if produces {
                if let Some(&rc) = rcs.get(&op_id) {
                    let dtype = dtypes[&op_id];
                    let idx = reg_slots
                        .iter()
                        .position(|(r, dt)| *r == 0 && *dt == dtype)
                        .unwrap_or_else(|| {
                            let i = reg_slots.len();
                            reg_slots.push((0, dtype));
                            i
                        });
                    reg_slots[idx].0 = rc;
                    reg_map.insert(op_id, idx);
                }
            }

            // Decrement RC for each operand
            for param in op.parameters() {
                if let Some(&p) = reg_map.get(&param) {
                    if reg_slots[p].0 > 0 {
                        reg_slots[p].0 -= 1;
                    }
                }
            }

            // Is this indexing or compute?
            if (matches!(op, Op::Index { .. } | Op::Loop { .. })
                || (op.parameters().count() > 0 && op.parameters().all(|p| indexing_ops.contains(&p))))
            {
                indexing_ops.insert(op_id);
            }
            if let Op::Const(c) = op {
                if c.dtype() == IDX_T {
                    indexing_ops.insert(op_id);
                }
            }

            // Instruction counting
            match op {
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
                | Op::Define { .. }
                | Op::EndIf
                | Op::Devectorize { .. }
                | Op::Vectorize { .. }
                | Op::ConstView(_)
                | Op::LoadView(_)
                | Op::StoreView { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => {}
                &Op::Load { src, index, vlen } => {
                    wi_ops += loop_mult;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult;
                    }
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    let total_elements = loop_mult * u64::from(vlen);
                    match scope {
                        Scope::Global => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size() as u64;
                            n_scoped_load_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Local, .. }) {
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
                                        if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Global, .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
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
                                glb_load_lidx_stride_weighted += 1 * n_bits;
                                glb_load_lidx_stride_weight += n_bits;
                            }
                        }
                        Scope::Local => {
                            let n_bits = total_elements * dtypes[&op_id].0.bit_size() as u64;
                            n_scoped_load_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Local, .. }) {
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
                                        if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Global, .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
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
                                loc_load_lidx_stride_weighted += 1 * n_bits;
                                loc_load_lidx_stride_weight += n_bits;
                            }
                        }
                        Scope::Register => {
                            n_scoped_load_bits[2] += total_elements * dtypes[&op_id].0.bit_size() as u64;
                        }
                    }
                }
                &Op::Store { dst, index, vlen, .. } => {
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    match scope {
                        Scope::Global => {
                            let n_bits = loop_mult * u64::from(vlen) * dtypes[&op_id].0.bit_size() as u64;
                            n_scoped_store_bits[0] += n_bits;
                            // Track stride: prefer lidx > gidx > loop
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Local, .. }) {
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
                                        if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Global, .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
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
                                glb_store_lidx_stride_weighted += 1 * n_bits;
                                glb_store_lidx_stride_weight += n_bits;
                            }
                        }
                        Scope::Local => {
                            let n_bits = loop_mult * u64::from(vlen) * dtypes[&op_id].0.bit_size() as u64;
                            n_scoped_store_bits[1] += n_bits;
                            let strides = self.get_strides(index);
                            let stride = strides
                                .iter()
                                .find_map(|(oid, (_, st))| {
                                    if oid.is_null() || *st == 0 {
                                        return None;
                                    }
                                    if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Local, .. }) {
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
                                        if matches!(self.ops[*oid].op, Op::Index { scope: Scope::Global, .. }) {
                                            Some(*st)
                                        } else {
                                            None
                                        }
                                    })
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
                                loc_store_lidx_stride_weighted += 1 * n_bits;
                                loc_store_lidx_stride_weight += n_bits;
                            }
                        }
                        Scope::Register => {
                            n_scoped_store_bits[2] += loop_mult * u64::from(vlen) * dtypes[&op_id].0.bit_size() as u64
                        }
                    }
                }
                &Op::Index { len, scope, axis } => match scope {
                    Scope::Global => gws[axis as usize] = len,
                    Scope::Local => lws[axis as usize] = len,
                    Scope::Register => {}
                },
                Op::Loop { len } => {
                    wi_ops += loop_mult * 3;
                    if !indexing_ops.contains(&op_id) {
                        wi_compute_ops += loop_mult * 3;
                    }
                    loop_mult *= *len as u64;
                    latest_loop_lengths.push(*len as u64);
                }
                Op::EndLoop => {
                    loop_mult /= latest_loop_lengths.pop().unwrap();
                }
                &Op::Wmma { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    let warp = u64::from(dev_info.warp_size);
                    let cost = (m * n * k) / warp;
                    wi_ops += loop_mult * cost;
                    if !indexing_ops.contains(&op_id) {
                        // TODO multiply by some constant
                        wi_compute_ops += loop_mult * cost;
                    }
                }
                Op::Barrier { .. } => {
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
            let bytes: u64 = reg_slots
                .iter()
                .filter(|(r, _)| *r > 0)
                .map(|(_, dt)| u64::from(dt.0.bit_size() / 8) * u64::from(dt.1))
                .sum();
            if bytes > wi_peak_reg_bytes {
                wi_peak_reg_bytes = bytes;
            }

            op_id = self.next_op(op_id);
        }

        let wi_global_load_bits = n_scoped_load_bits[0];
        let wi_local_load_bits = n_scoped_load_bits[1];
        let wi_global_store_bits = n_scoped_store_bits[0];
        let wi_local_store_bits = n_scoped_store_bits[1];

        let num_groups = gws.iter().product::<u64>();
        let wi_per_group = lws.iter().product::<u64>();

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

        // Learned cost model: rank 0..1 within variant * 1_000_000 (2000 DT leaves + Ridge, ρ=0.93)
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
        );
        let cost = cost.max(1.0) as u64;

        Cost {
            cost,
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
            wi_global_load_lidx_stride: if glb_load_lidx_stride_weight > 0 {
                (glb_load_lidx_stride_weighted as f64 / glb_load_lidx_stride_weight as f64 * 10.0) as u64
            } else {
                0
            },
            wi_global_store_lidx_stride: if glb_store_lidx_stride_weight > 0 {
                (glb_store_lidx_stride_weighted as f64 / glb_store_lidx_stride_weight as f64 * 10.0) as u64
            } else {
                0
            },
            wi_local_load_lidx_stride: if loc_load_lidx_stride_weight > 0 {
                (loc_load_lidx_stride_weighted as f64 / loc_load_lidx_stride_weight as f64 * 10.0) as u64
            } else {
                0
            },
            wi_local_store_lidx_stride: if loc_store_lidx_stride_weight > 0 {
                (loc_store_lidx_stride_weighted as f64 / loc_store_lidx_stride_weight as f64 * 10.0) as u64
            } else {
                0
            },
            warp_size: dev_info.warp_size as u64,
            max_local_threads: dev_info.max_local_threads,
            max_register_bytes: dev_info.max_register_bytes,
        }
    }
}
