// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map,
    backend::DeviceInfo,
    kernel::{Kernel, Op, OpId, Scope},
};
use nanoserde::{DeBin, SerBin};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, DeBin, SerBin)]
pub struct Cost {
    pub cost: u64,
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
        let mut n_instructions = 0;
        let mut n_mads = 0u64;
        let mut n_scoped_loads = [0u64; 3];
        let mut n_scoped_stores = [0u64; 3];
        let mut reuse_credit = [0u64; 3];
        let mut barriers_per_thread = 0u64;
        let mut gws = [1u64; 3];
        let mut lws = [1u64; 3];
        let mut loop_mult = 1u64;
        let mut latest_loop_lengths: Vec<u64> = Vec::new();

        let mut reg_slots: Vec<(u32, (DType, u16))> = Vec::new(); // (rc, dtype)
        let mut reg_map: Map<OpId, usize> = Map::default();
        let mut peak_reg_bytes = 0u64;

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
                | Op::Wmma { .. } => true,
                _ => false,
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
            match op {
                &Op::Load { index, .. } => {
                    if let Some(&idx) = reg_map.get(&index) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                &Op::Store { x, index, .. } => {
                    if let Some(&idx) = reg_map.get(&x) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&index) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                &Op::Cast { x, .. } | &Op::Unary { x, .. } => {
                    if let Some(&idx) = reg_map.get(&x) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                &Op::Binary { x, y, .. } => {
                    if let Some(&idx) = reg_map.get(&x) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&y) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                &Op::Mad { x, y, z } => {
                    if let Some(&idx) = reg_map.get(&x) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&y) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&z) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                Op::Vectorize { ops } => {
                    for &x in ops.iter() {
                        if let Some(&idx) = reg_map.get(&x) {
                            if reg_slots[idx].0 > 0 {
                                reg_slots[idx].0 -= 1;
                            }
                        }
                    }
                }
                &Op::Wmma { a, b, c, .. } => {
                    if let Some(&idx) = reg_map.get(&a) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&b) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                    if let Some(&idx) = reg_map.get(&c) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                &Op::If { condition } => {
                    if let Some(&idx) = reg_map.get(&condition) {
                        if reg_slots[idx].0 > 0 {
                            reg_slots[idx].0 -= 1;
                        }
                    }
                }
                _ => {}
            }

            // Instruction counting (original logic)
            match op {
                Op::Cast { .. } | Op::Unary { .. } | Op::Binary { .. } => {
                    n_instructions += loop_mult;
                }
                Op::Mad { .. } => {
                    n_instructions += loop_mult;
                    n_mads += loop_mult;
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
                &Op::Load { src, vlen, .. } => {
                    n_instructions += loop_mult;
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    let total_elements = loop_mult * u64::from(vlen);
                    let rc = rcs.get(&op_id).copied().unwrap_or(1);
                    let extra_uses = rc.saturating_sub(1) as u64;
                    let credit = total_elements * extra_uses;
                    match scope {
                        Scope::Global => {
                            n_scoped_loads[0] += total_elements;
                            reuse_credit[0] += credit;
                        }
                        Scope::Local => {
                            n_scoped_loads[1] += total_elements;
                            reuse_credit[1] += credit;
                        }
                        Scope::Register => {
                            n_scoped_loads[2] += total_elements;
                            reuse_credit[2] += credit;
                        }
                    }
                }
                &Op::Store { dst, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    match scope {
                        Scope::Global => n_scoped_stores[0] += loop_mult * u64::from(vlen),
                        Scope::Local => n_scoped_stores[1] += loop_mult * u64::from(vlen),
                        Scope::Register => n_scoped_stores[2] += loop_mult * u64::from(vlen),
                    }
                }
                &Op::Index { len, scope, axis } => match scope {
                    Scope::Global => gws[axis as usize] = len,
                    Scope::Local => lws[axis as usize] = len,
                    Scope::Register => {}
                },
                Op::Loop { len } => {
                    n_instructions += loop_mult * 3;
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
                    n_instructions += loop_mult * cost;
                }
                Op::Barrier { .. } => {
                    barriers_per_thread += loop_mult;
                }
                Op::If { .. } => {
                    n_instructions += loop_mult * 3;
                }
            }

            // Track peak register bytes
            let bytes: u64 = reg_slots
                .iter()
                .filter(|(r, _)| *r > 0)
                .map(|(_, dt)| u64::from(dt.0.bit_size() / 8) * u64::from(dt.1))
                .sum();
            if bytes > peak_reg_bytes {
                peak_reg_bytes = bytes;
            }

            op_id = self.next_op(op_id);
        }

        let n_threads = lws.iter().product::<u64>();
        let global_load_ops_per_thread = n_scoped_loads[0].saturating_sub(reuse_credit[0]);
        let local_loads_per_thread = n_scoped_loads[1];
        let global_stores_per_thread = n_scoped_stores[0];
        let local_stores_per_thread = n_scoped_stores[1];

        let global_ws = gws.iter().product::<u64>();
        // Subtract reuse credit: values loaded once and used by multiple
        // consumers (register reuse) should not be charged multiple times.
        let per_thread_loads = global_ws * global_load_ops_per_thread;
        let per_thread_stores = global_ws * global_stores_per_thread;
        let per_thread_instr = global_ws * n_instructions;
        let workgroup_local =
            n_threads * global_ws * (local_loads_per_thread + local_stores_per_thread);
        let workgroup_barriers = n_threads * global_ws * barriers_per_thread;

        let memory_score = (per_thread_loads * 10 + per_thread_stores * 10 + workgroup_local
            + workgroup_barriers * 20) as f64;
        let alu_score = per_thread_instr as f64;

        // Use actual loaded elements (with vlen) for bandwidth / arith intensity.
        let bytes_total = (global_ws * n_scoped_loads[0] + per_thread_stores) * 4;
        let flops_total = n_mads * 2 * global_ws;
        let arith_intensity = if bytes_total > 0 {
            flops_total as f64 / bytes_total as f64
        } else {
            0.0
        };
        // Penalize register pressure relative to device limit.
        let reg_pressure = peak_reg_bytes as f64;
        let max_regs = dev_info.max_register_bytes as f64;

        let cost = (memory_score + alu_score * 0.1 - arith_intensity * 5.0
            + if reg_pressure > max_regs { 1_000_000.0 } else { 0.0 }) as u64;

        Cost { cost }
    }
}
