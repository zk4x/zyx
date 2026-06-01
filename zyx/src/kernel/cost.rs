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

    warp_size: u64,
    max_local_threads: u64,
    max_register_bytes: u64,
}

impl Cost {
    pub fn debug(&self) {
        println!(
            "cost={}, num_groups={}, wi_per_group={}, wi_ops={}, wi_compute_ops={}, wi_barriers={}, wi_global_load_bits={}, wi_global_store_bits={}
wi_local_load_bits={}, wi_local_store_bits={}, wi_peak_reg_bytes={}, wi_branches={}, wi_global_load_lidx_stride={}, wi_global_store_lidx_stride={}, warp_size={}, max_local_threads={}, max_register_bytes={}",
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
                            n_scoped_load_bits[1] += total_elements * dtypes[&op_id].0.bit_size() as u64;
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
                            n_scoped_store_bits[1] += loop_mult * u64::from(vlen) * dtypes[&op_id].0.bit_size() as u64
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

        // Learned cost model (Lasso on OpenCL autotune data, 39 features, R²=0.949)
        // Predicts log(estimated_time_us), then exponentiates.
        let gmem = (wi_global_load_bits + wi_global_store_bits) as f64;
        let local_mem = (wi_local_load_bits + wi_local_store_bits) as f64;
        let reg_ratio = wi_peak_reg_bytes as f64 / dev_info.max_register_bytes as f64;
        let warp_ratio = wi_per_group as f64 / dev_info.warp_size as f64;
        let ci = wi_compute_ops as f64 / gmem.max(1.0);

        let lng = (num_groups.max(1) as f64).ln();
        let lwpg = (wi_per_group.max(1) as f64).ln();
        let lops = (wi_ops.max(1) as f64).ln();
        let lcop = (wi_compute_ops.max(1) as f64).ln();
        let lgmem = gmem.max(1.0).ln();
        let opt = wi_ops as f64 / (num_groups.max(1) as f64 * wi_per_group.max(1) as f64);

        let lld_st = glb_load_lidx_stride.max(0.0);
        let lst_st = glb_store_lidx_stride.max(0.0);

        let b0 = if wi_barriers == 0 { 1.0 } else { 0.0 };
        let b3 = if wi_barriers == 3 { 1.0 } else { 0.0 };
        let b4 = if wi_barriers == 4 { 1.0 } else { 0.0 };
        let b7 = if wi_barriers == 7 { 1.0 } else { 0.0 };
        let barr = wi_barriers as f64;
        let total_threads = (num_groups * wi_per_group) as f64;
        let raw_ng = num_groups as f64;

        let log_lwpg = lwpg.max(1e-8).ln();
        let lng_log_opt = lng * opt.ln_1p();
        let lng_log_lwpg = lng * log_lwpg;
        let lwpg_log_opt = lwpg * opt.ln_1p();
        let lng_log1p_lwpg_div_opt = lng * (1.0 + lwpg / opt.max(1e-8)).ln();

        // 39 selected features from Lasso
        let log_time_us =
            0.035030 * barr + 0.015315 * total_threads + 0.013025 * (wi_global_store_bits as f64 / wi_per_group.max(1) as f64)
                - 0.062212 * (gmem / total_threads.max(1.0))
                - 0.056089 * ci.ln_1p()
                - 0.035205 * (wi_ops as f64 / wi_compute_ops.max(1) as f64).ln_1p()
                - 0.131589 * (100.0 / wi_ops.max(1) as f64).ln_1p()
                + 0.376882 * (1000.0 / wi_compute_ops.max(1) as f64).ln_1p()
                + 0.178002 * log_lwpg
                - 0.484077 * lng_log_opt
                - 0.164596 * lng_log_lwpg
                - 0.075348 * lwpg_log_opt
                + 0.046618 * raw_ng
                + 0.142005 * (num_groups * wi_per_group) as f64
                - 0.078469 * raw_ng / wi_ops.max(1) as f64
                + 0.030256 * (wi_branches as f64).ln_1p()
                - 0.038757 * lng * lwpg
                + 0.998624 * lng_log1p_lwpg_div_opt
                - 0.011622 * b0
                + 0.031162 * b4
                + 0.000044 * b7
                + 0.013697 * b3 * lng
                - 0.012228 * b4 * lng
                - 0.007321 * b7 * lng
                - 0.051676 * barr * lng
                + 0.046386 * lld_st.ln_1p() * warp_ratio
                + 0.007545 * lld_st.ln_1p() * lst_st.ln_1p()
                + 0.029707 * lst_st.ln_1p() * lng
                + 2.685514 * lops * lgmem
                + 0.179980 * lcop * lgmem
                + 0.098591 * lwpg * lgmem
                + 0.159286 * lcop * ci
                - 0.043359 * lng * ci
                - 0.019263 * lld_st.ln_1p() * ci
                + 0.100067 * lwpg * reg_ratio
                + 0.901502 * lng * lcop
                - 0.051377 * lwpg * (32.0 / wi_per_group.max(1) as f64).ln()
                + 0.041226 * raw_ng * lwpg
                - 0.010186 * barr * (1.0 + num_groups as f64 / local_mem.max(1.0)).ln()
                + 5.755699;

        let cost = log_time_us.exp().max(1.0) as u64;

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
            warp_size: dev_info.warp_size as u64,
            max_local_threads: dev_info.max_local_threads,
            max_register_bytes: dev_info.max_register_bytes,
        }
    }
}
