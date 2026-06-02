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

        // Learned cost model: 20 Ridge features + 80-leaf DT (100 params, R²=0.931 MatMul, 0.818 worst)
        let cost = Self::predict_log_time(
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
        )
        .max(1.0) as u64;

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

    fn predict_log_time(
        num_groups: u32,
        wi_per_group: u32,
        wi_ops: u32,
        wi_compute_ops: u32,
        wi_barriers: u32,
        wi_global_load_bits: u32,
        wi_global_store_bits: u32,
        wi_local_load_bits: u32,
        wi_local_store_bits: u32,
        wi_peak_reg_bytes: u32,
        _wi_branches: u32,
        wi_global_load_lidx_stride: u32,
        wi_global_store_lidx_stride: u32,
        _wi_local_load_lidx_stride: u32,
        _wi_local_store_lidx_stride: u32,
        warp_size: u32,
        _max_local_threads: u32,
        max_register_bytes: u32,
    ) -> f32 {
        let lng = (num_groups as f32).ln();
        let lwpg = (wi_per_group as f32 + 1.0).ln();
        let lops = (wi_ops as f32).ln();
        let lcop = (wi_compute_ops as f32).ln();
        let lgmem = ((wi_global_load_bits + wi_global_store_bits) as f32 + 1.0).ln();
        let _log_inv_threads = (1.0 / (num_groups * wi_per_group) as f32).ln_1p();
        let ci = wi_compute_ops as f32 / ((wi_global_load_bits + wi_global_store_bits) as f32).max(1.0);
        let _barr = wi_barriers as f32;
        let wr = wi_per_group as f32 / warp_size as f32;
        let rr = wi_peak_reg_bytes as f32 / max_register_bytes.max(1) as f32;

        let features: [f32; 20] = [
            lwpg,
            lcop,
            lgmem,
            (wi_ops as f32 + 10.0).ln(),
            wi_ops as f32 / (num_groups * wi_per_group).max(1) as f32,
            wi_ops as f32 / num_groups.max(1) as f32,
            lwpg.max(1e-8).ln(),
            (wi_ops as f32 / (num_groups * wi_per_group).max(1) as f32).ln_1p(),
            (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lwpg,
            lops * lgmem,
            (32.0 / wi_per_group.max(1) as f32).ln(),
            (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lcop,
            (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lgmem,
            wi_compute_ops as f32 / num_groups.max(1) as f32,
            (32.0 / wi_per_group.max(1) as f32).max(1e-8).ln(),
            wi_compute_ops as f32 / (num_groups * wi_barriers.max(1)).max(1) as f32,
            wi_ops as f32 / (num_groups as f32 * (wi_per_group as f32).ln_1p()).max(1.0),
            (lwpg.abs()).ln_1p(),
            (if wi_barriers > 0 { 1.0 } else { 0.0 }) * lcop,
            (if wi_barriers > 0 { 1.0 } else { 0.0 }) * lgmem,
        ];

        // DT leaf bias
        let leaf_bias: f32 = {
            if lcop < 14.6954f32 {
                if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln() < 85.1066f32 {
                    if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lgmem < 13.1727f32 {
                        if ((num_groups * wi_per_group) as f32) < 1.31072e+06f32 {
                            if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln() < 61.9768f32 {
                                if lgmem < 11.792f32 {
                                    if (lng.abs()).ln_1p() < 2.45085f32 {
                                        if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                            < 55.7973f32
                                        {
                                            if (wi_global_load_lidx_stride as f32).ln_1p() * lgmem < 7.21556f32 {
                                                if lng < 9.21365f32 {
                                                    if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lops < 2.602f32 {
                                                        -0.315042
                                                    } else {
                                                        if ((num_groups * wi_per_group) as f32) < 14336.0f32 {
                                                            -0.314046
                                                        } else {
                                                            -0.149670
                                                        }
                                                    }
                                                } else {
                                                    -0.087817
                                                }
                                            } else {
                                                if lwpg * rr < 0.258709f32 {
                                                    -0.239622
                                                } else {
                                                    if rr < 3.28906f32 { -0.257033 } else { -0.037636 }
                                                }
                                            }
                                        } else {
                                            if (num_groups as f32 / wi_ops.max(1) as f32 + 1.0).ln() < 3.98632f32 {
                                                if ((wi_local_load_bits + wi_local_store_bits + 1) as f32
                                                    / (wi_global_load_bits + wi_global_store_bits + 1).max(1) as f32)
                                                    .ln_1p()
                                                    < 6.03236e-05f32
                                                {
                                                    -0.203952
                                                } else {
                                                    if lcop * ci < 1.05376f32 { -0.089467 } else { -0.232103 }
                                                }
                                            } else {
                                                0.016597
                                            }
                                        }
                                    } else {
                                        if (lng.abs()).ln_1p() < 2.57456f32 {
                                            -0.005460
                                        } else {
                                            0.069694
                                        }
                                    }
                                } else {
                                    if lcop * lgmem < 134.933f32 {
                                        if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                            < 47.6453f32
                                        {
                                            -0.132553
                                        } else {
                                            -0.060170
                                        }
                                    } else {
                                        if lwpg * lops * (if wi_barriers == 0 { 1.0 } else { 0.0 }) < 25.9433f32 {
                                            if (wi_compute_ops as f32 + 1.0).ln()
                                                * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                                < 47.2321f32
                                            {
                                                -0.060496
                                            } else {
                                                -0.016397
                                            }
                                        } else {
                                            0.027505
                                        }
                                    }
                                }
                            } else {
                                if lng * lgmem < 88.1638f32 {
                                    if lwpg * rr < 2.95344f32 {
                                        if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                            < 74.1777f32
                                        {
                                            if lng * lgmem < 81.5644f32 {
                                                if (wi_compute_ops as f32 + 1.0).ln()
                                                    * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                                    < 66.9711f32
                                                {
                                                    if (wi_ops as f32 / num_groups.max(1) as f32 + 1.0).ln() < 0.0177702f32 {
                                                        0.032813
                                                    } else {
                                                        if lng * rr < 5.1607f32 { -0.139093 } else { -0.052239 }
                                                    }
                                                } else {
                                                    -0.013614
                                                }
                                            } else {
                                                if (wi_ops as f32
                                                    / (num_groups * wi_per_group * wi_barriers.max(1)).max(1) as f32)
                                                    .ln_1p()
                                                    < 0.00137616f32
                                                {
                                                    0.059518
                                                } else {
                                                    -0.026348
                                                }
                                            }
                                        } else {
                                            if lops * ci < 0.635683f32 { 0.037472 } else { 0.045831 }
                                        }
                                    } else {
                                        if ((num_groups * wi_per_group) as f32) < 81920.0f32 {
                                            if wi_ops as f32 / (num_groups as f32 * (wi_per_group as f32).ln_1p()).max(1.0)
                                                < 52.3908f32
                                            {
                                                if (wi_compute_ops as f32 + 1.0).ln()
                                                    * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                                    < 73.7971f32
                                                {
                                                    if (wi_global_load_lidx_stride as f32).ln_1p() * lops < 5.80266f32 {
                                                        -0.044590
                                                    } else {
                                                        -0.066834
                                                    }
                                                } else {
                                                    if lops * ci < 0.995892f32 { 0.038506 } else { -0.003746 }
                                                }
                                            } else {
                                                if lng * lgmem < 34.6903f32 { 0.053176 } else { -0.008790 }
                                            }
                                        } else {
                                            0.080890
                                        }
                                    }
                                } else {
                                    if ((num_groups * wi_per_group) as f32) < 1.6384e+05f32 {
                                        if lng * lgmem < 95.1613f32 { 0.067617 } else { 0.090148 }
                                    } else {
                                        if (wi_global_store_lidx_stride as f32).ln_1p() < 12.3005f32 {
                                            0.111402
                                        } else {
                                            0.065163
                                        }
                                    }
                                }
                            }
                        } else {
                            if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lng < 14.0657f32 {
                                if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lops < 2.86179f32 {
                                    if lng * lgmem < 51.4088f32 { 0.002436 } else { 0.079411 }
                                } else {
                                    if ((num_groups * wi_per_group) as f32) < 1.04858e+07f32 {
                                        if ((wi_local_load_bits + wi_local_store_bits + 1) as f32
                                            / (wi_global_load_bits + wi_global_store_bits + 1).max(1) as f32)
                                            .ln_1p()
                                            < 0.00644501f32
                                        {
                                            if lng * lcop < 57.8961f32 { 0.188530 } else { 0.109664 }
                                        } else {
                                            0.041325
                                        }
                                    } else {
                                        if lwpg * lgmem < 21.9103f32 { 0.171483 } else { 0.208951 }
                                    }
                                }
                            } else {
                                if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lng < 15.452f32 {
                                    0.269930
                                } else {
                                    0.269084
                                }
                            }
                        }
                    } else {
                        if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lgmem < 14.5568f32 {
                            if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lgmem < 14.2694f32 {
                                if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln()
                                    < 54.9967f32
                                {
                                    if lwpg * lops < 34.8776f32 {
                                        if (wi_ops as f32 / wi_compute_ops.max(1) as f32).ln_1p() < 0.899454f32 {
                                            if wi_ops as f32 / (wi_barriers.max(1) as f32) < 2.06145e+05f32 {
                                                -0.015486
                                            } else {
                                                0.040021
                                            }
                                        } else {
                                            0.033981
                                        }
                                    } else {
                                        0.066556
                                    }
                                } else {
                                    0.083887
                                }
                            } else {
                                if ((wi_global_load_bits + wi_global_store_bits) as f32
                                    / (num_groups * wi_per_group).max(1) as f32)
                                    .ln_1p()
                                    < 11.7838f32
                                {
                                    0.140588
                                } else {
                                    0.075589
                                }
                            }
                        } else {
                            if (if wi_barriers == 0 { 1.0 } else { 0.0 }) * lgmem < 15.367f32 {
                                if ((num_groups * wi_per_group) as f32 / wi_ops.max(1) as f32 + 1.0).ln() < 0.000203224f32 {
                                    0.136688
                                } else {
                                    0.079737
                                }
                            } else {
                                0.305419
                            }
                        }
                    }
                } else {
                    if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln() < 117.9f32 {
                        if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln() < 93.7339f32 {
                            if wi_ops as f32 / (num_groups as f32 * (wi_per_group as f32).ln_1p()).max(1.0) < 0.0062929f32 {
                                0.240872
                            } else {
                                if lcop * ci < 1.05826f32 {
                                    if wi_compute_ops as f32 / ((num_groups * wi_barriers.max(1)).max(1) as f32) < 9.11719f32 {
                                        0.167117
                                    } else {
                                        0.088937
                                    }
                                } else {
                                    if rr < 5.32031f32 {
                                        if ((num_groups * wi_per_group) as f32 / wi_ops.max(1) as f32 + 1.0).ln() < 4.18802f32 {
                                            if lwpg * lgmem < 36.195f32 { 0.120424 } else { 0.037685 }
                                        } else {
                                            0.128008
                                        }
                                    } else {
                                        0.093967
                                    }
                                }
                            }
                        } else {
                            if rr < 2.53906f32 {
                                if lng * wr < 11.7835f32 {
                                    if lng * lgmem < 123.103f32 {
                                        if (wi_global_store_lidx_stride as f32).ln_1p() < 0.346574f32 {
                                            if lng * lwpg < 24.302f32 { 0.057578 } else { 0.050432 }
                                        } else {
                                            0.218295
                                        }
                                    } else {
                                        0.154045
                                    }
                                } else {
                                    if ci < 0.187576f32 {
                                        0.057914
                                    } else {
                                        if (wi_global_load_lidx_stride as f32).ln_1p() * lcop < 4.98745f32 {
                                            0.085981
                                        } else {
                                            0.126168
                                        }
                                    }
                                }
                            } else {
                                if ((wi_global_load_bits + wi_global_store_bits) as f32 / wi_compute_ops.max(1) as f32 + 1.0).ln()
                                    < 1.71697f32
                                {
                                    0.061721
                                } else {
                                    if wi_ops as f32 / (num_groups as f32 * (wi_per_group as f32).ln_1p()).max(1.0) < 38.0049f32 {
                                        if lng * rr < 53.9843f32 { 0.120308 } else { 0.165767 }
                                    } else {
                                        0.186418
                                    }
                                }
                            }
                        }
                    } else {
                        if lng * lgmem < 151.85f32 {
                            if (wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln() < 143.821f32 {
                                0.226339
                            } else {
                                0.158249
                            }
                        } else {
                            0.284622
                        }
                    }
                }
            } else {
                if (100.0 / wi_ops.max(1) as f32).ln_1p() < 2.8236e-07f32 {
                    0.596516
                } else {
                    if rr < 9.14062f32 { 0.490526 } else { 0.032389 }
                }
            }
        };

        let mut result = 5.684563;
        result += -23.816857 * (features[0] - 1.874778) / 1.409477;
        result += -0.331053 * (features[1] - 8.965733) / 3.008445;
        result += -0.003982 * (features[2] - 10.867630) / 2.972404;
        result += 0.219753 * (features[3] - 9.334600) / 2.910882;
        result += 88.788546 * (features[4] - 10325367.092181) / 115660116.274095;
        result += -16.302304 * (features[5] - 10332302.250345) / 115659516.621871;
        result += -17.480788 * (features[6] - 0.363830) / 0.719155;
        result += -0.363476 * (features[7] - 4.033228) / 4.230333;
        result += -0.102154 * (features[8] - 1.617893) / 1.403302;
        result += 0.784724 * (features[9] - 109.898876) / 64.158217;
        result += -14.112917 * (features[10] - 1.958964) / 1.668577;
        result += -0.617851 * (features[11] - 8.453127) / 3.789894;
        result += 0.437918 * (features[12] - 10.229378) / 4.094774;
        result += -133.041843 * (features[13] - 6970149.690773) / 77210567.047053;
        result += -14.112916 * (features[14] - 1.958964) / 1.668577;
        result += 133.001714 * (features[15] - 6970012.281720) / 77210579.416615;
        result += -72.509976 * (features[16] - 10331095.199491) / 115659621.780240;
        result += 12.805095 * (features[17] - 0.951372) / 0.443286;
        result += 0.734244 * (features[18] - 0.512606) / 1.831308;
        result += -0.797162 * (features[19] - 0.638252) / 2.264033;
        result += leaf_bias;
        result
    }
}
