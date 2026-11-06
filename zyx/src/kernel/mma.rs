// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Automatic tensorization: turn a scalar matmul accumulation loop into a
//! warp-level matrix multiply (`Op::Wmma`).
//!
//! The input is the linearized scalar form — one output element per thread,
//! a K loop accumulating `acc = a[m,k] * b[k,n] + acc`:
//!
//! ```text
//! store acc = 0
//! for k in 0..K { acc = A[m*K + k] * B[k*N + n] + acc }
//! store C[m*N + n] = acc
//! ```
//!
//! Configs are one per device instruction variant the IR fits: the operand and
//! accumulator dtypes must match the variant's, and every extent must be a
//! whole tile of it (no padding yet). The table lives in
//! [`crate::backend::DeviceInfo::mma_variants`], defined by each backend —
//! hardware description, not kernel logic. The matcher only inspects the IR,
//! nothing is mutated at make time, so `apply` sees exactly the IR the config
//! was derived from.
//!
//! `apply` performs the tiling and the emission itself: it narrows the M/N
//! group ranges to the variant's tile grid, emits the warp lane, builds each
//! fragment slot's address from the variant's lane bit-fields, runs the K loop
//! one instruction per `k` step, and scatters the accumulator slots. All of it
//! is scalar IR (slot-wise register accesses, no vector lanes), so the device
//! epilogue has nothing to devectorize in a tensorized kernel. The scalar
//! region (init store, loop, final store) is removed — `dead_code_elimination`
//! keeps every `Store` and `Loop` as a root, so it cannot remove them on its
//! own.

use crate::{
    Set,
    backend::{FragAxis, FragMap, MMAVariant},
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, RangeKind, autotune::Optimization},
    shape::Dim,
};

/// One scalar matmul region and the instruction variant it will be built
/// with: the accumulator's init store, the final store of the sum, and the
/// affine views of all three operands.
#[derive(Debug, Clone)]
struct Region {
    variant: MMAVariant,
    init_store: OpId,
    final_store: OpId,

    a_src: OpId,
    a_m: Dim,
    a_k: Dim,
    b_src: OpId,
    b_n: Dim,
    b_k: Dim,
    c_src: OpId,
    c_m: Dim,
    c_n: Dim,

    m_range: OpId,
    n_range: OpId,
    m: Dim,
    n: Dim,
    k: Dim,
}

/// The tensorization optimization: one config per validated region.
#[derive(Debug)]
struct Tensorize {
    regions: Vec<Region>,
}

impl Optimization for Tensorize {
    fn nconfigs(&self) -> u64 {
        self.regions.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        let Some(r) = self.regions.get(config as usize) else {
            return;
        };
        let v = &r.variant;
        let (fm, fn_, fk) = {
            let (m, n, k) = v.dims.decompose_mnk();
            (m as Dim, n as Dim, k as Dim)
        };
        let warp_size = u32::from(kernel.device_info().warp_size);

        // Narrow the M/N iteration to the fragment grid: a block now computes
        // one whole tile instead of one element.
        {
            for (range, len) in [(r.m_range, r.m / fm), (r.n_range, r.n / fn_)] {
                let len = kernel.insert_before(range, Op::Const(Constant::idx(len)));
                let Op::Range { axis, .. } = kernel.at(range) else {
                    unreachable!()
                };
                let axis = *axis;
                kernel.ops[range].op = Op::Range { axis, kind: RangeKind::Group(len) };
            }
        }

        // The warp lane (local axis 0) and this block's tile origin.
        let lane = kernel.local_range(0, warp_size);
        let fm_c = kernel.const_idx(fm);
        let m_base = kernel.mul(r.m_range, fm_c);
        let fn_c = kernel.const_idx(fn_);
        let n_base = kernel.mul(r.n_range, fn_c);

        // Element index of one fragment slot of an operand: the lane's
        // `(row, col)` pair from the variant's lane bit-fields, plus the
        // slot's own offset, placed at the block's `(base_row, base_col)` in
        // the view's coordinate space and strided by the view's strides.
        //
        // A field reads `width` bits of the lane id starting at `shift` and
        // adds `stride` per step — divisor arithmetic, `lane / 2^shift` (and
        // `% 2^width` when the field is narrower than the warp), which is how
        // every documented fragment layout is described. Nothing here knows
        // which instruction it is emitting for; the table in `DeviceInfo` is
        // the whole description.
        let frag_index = |kernel: &mut Kernel,
                          map: &FragMap,
                          (base_row, base_col): (OpId, OpId),
                          (stride_row, stride_col): (Dim, Dim),
                          slot: (i16, i16)|
         -> OpId {
            let axis_sum = |kernel: &mut Kernel, axis: FragAxis| -> OpId {
                let mut sum = kernel.const_idx(0u32);
                for field in map.fields.iter().filter(|f| f.axis == axis) {
                    let mut term = lane;
                    if field.shift > 0 {
                        let d = kernel.const_idx(1u32 << field.shift);
                        term = kernel.div(term, d);
                    }
                    if (1u32 << field.width) < warp_size {
                        let w = kernel.const_idx(1u32 << field.width);
                        term = kernel.mod_(term, w);
                    }
                    if field.stride != 1 {
                        let s = kernel.const_idx(i64::from(field.stride));
                        term = kernel.mul(term, s);
                    }
                    sum = kernel.add(sum, term);
                }
                sum
            };
            let row = axis_sum(kernel, FragAxis::Row);
            let col = axis_sum(kernel, FragAxis::Col);
            let row = kernel.add(row, base_row);
            let col = kernel.add(col, base_col);
            let row = if slot.0 == 0 {
                row
            } else {
                let off = kernel.const_idx(i64::from(slot.0));
                kernel.add(row, off)
            };
            let col = if slot.1 == 0 {
                col
            } else {
                let off = kernel.const_idx(i64::from(slot.1));
                kernel.add(col, off)
            };
            let col = if stride_col == 1 {
                col
            } else {
                let s = kernel.const_idx(stride_col);
                kernel.mul(col, s)
            };
            let s = kernel.const_idx(stride_row);
            kernel.mad(row, s, col)
        };

        // The accumulator: one register per fragment slot, zero-initialized,
        // read and written scalar-wise (no vector lanes -> nothing for the
        // device epilogue to devectorize in this kernel).
        let c_slots = v.c.slots.len();
        let acc = kernel.zeros(v.c_dtype, c_slots as i32);

        // One mma per `fk` of K.
        let rounds = r.k / fk;
        kernel.loop_over(rounds, |k, kv| {
            let fk_c = k.const_idx(fk);
            let k_base = k.mul(kv, fk_c);
            let mut a_elems = Vec::with_capacity(v.a.slots.len());
            for &slot in &v.a.slots {
                let idx = frag_index(k, &v.a, (m_base, k_base), (r.a_m, r.a_k), slot);
                a_elems.push(k.load(r.a_src, idx));
            }
            let a_frag = k.stack(&a_elems);
            let mut b_elems = Vec::with_capacity(v.b.slots.len());
            for &slot in &v.b.slots {
                let idx = frag_index(k, &v.b, (n_base, k_base), (r.b_n, r.b_k), slot);
                b_elems.push(k.load(r.b_src, idx));
            }
            let b_frag = k.stack(&b_elems);
            let mut c_elems = Vec::with_capacity(c_slots);
            for i in 0..c_slots {
                let at = k.const_idx(i as u32);
                c_elems.push(k.load(acc, at));
            }
            let c_frag = k.stack(&c_elems);
            let d = k.wmma(v.dims, v.layout, v.dtype, a_frag, b_frag, c_frag);
            for i in 0..c_slots {
                let at = k.const_idx(i as u32);
                let elem = k.index(d, i);
                k.store(acc, elem, at);
            }
        });

        // Every store after the loop whose value traces back to the
        // accumulator is a C output: the matched final store plus any
        // duplicate of the same value (e.g. `f32(f16(acc))`, a cast copy that
        // is still live elsewhere). Each is scattered per fragment slot with
        // its own cast chain. The update store inside the loop never matches:
        // its source is the `add`, not a cast of an accumulator load.
        // The seed's accumulator: the init store's destination. `acc` below is
        // the fresh fragment storage `apply` just built.
        let old_acc = match kernel.at(r.init_store) {
            Op::Store { dst, .. } => *dst,
            _ => unreachable!("tensorize: init store is not a store"),
        };
        let mut epilogue: Vec<(OpId, Vec<crate::DType>)> = Vec::new();
        let mut extra_stores: Vec<OpId> = Vec::new();
        {
            let mut op_id = kernel.head;
            while !op_id.is_null() {
                if let Op::Store { dst, src, .. } = kernel.at(op_id) {
                    let mut chain: Vec<crate::DType> = Vec::new();
                    let mut value = *src;
                    loop {
                        match kernel.at(value) {
                            Op::Cast { x, dtype } | Op::Bitcast { x, dtype } => {
                                chain.push(*dtype);
                                value = *x;
                            }
                            Op::Load { src: s, .. } if *s == old_acc => {
                                if op_id != r.final_store {
                                    extra_stores.push(op_id);
                                }
                                epilogue.push((*dst, chain));
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                op_id = kernel.next_op(op_id);
            }
        }
        // Scatter the accumulator: each lane writes its own fragment slots,
        // once per C output.
        for (dst, chain) in &epilogue {
            for i in 0..c_slots {
                let at = kernel.const_idx(i as u32);
                let mut elem = kernel.load(acc, at);
                // `chain` runs outermost-first; rebuild the value inside-out.
                for dtype in chain.iter().rev() {
                    elem = kernel.cast(elem, *dtype);
                }
                let idx = frag_index(kernel, &v.c, (m_base, n_base), (r.c_m, r.c_n), v.c.slots[i]);
                kernel.store(*dst, elem, idx);
            }
        }

        // Drop the scalar region: `Store` and `Loop` are DCE roots, so the
        // old accumulation never dies on its own. `final_store` is inside the
        // span; any duplicate epilogue store sits after it and is removed on
        // its own.
        let mut doomed: Vec<OpId> = Vec::new();
        let mut op_id = r.init_store;
        while !op_id.is_null() {
            doomed.push(op_id);
            if op_id == r.final_store {
                break;
            }
            op_id = kernel.next_op(op_id);
        }
        for op_id in doomed.into_iter().rev() {
            kernel.remove_op(op_id);
        }
        for op_id in extra_stores.into_iter().rev() {
            kernel.remove_op(op_id);
        }
        kernel.verify();
    }
}

impl Kernel {
    /// Make the tensorization optimization: scan the linearized IR for
    /// scalar matmul regions, one config per region the device can
    /// accelerate. Never mutates.
    pub fn opt_tensorize(&self) -> Box<dyn Optimization> {
        let mut regions = Vec::new();
        let info = self.device_info();
        {
            // Device gate: a tensor-core unit offering at least one
            // instruction variant.
            if !info.tensor_cores || info.mma_variants.is_empty() {
                return Box::new(Tensorize { regions });
            }
        }

        // IR order once: the scan needs a lookback (the accumulator's init
        // store) and a lookahead (the store after the loop).
        let mut order: Vec<OpId> = Vec::new();
        {
            let mut op_id = self.head;
            while !op_id.is_null() {
                order.push(op_id);
                op_id = self.next_op(op_id);
            }
        }

        let mut seen: Set<OpId> = Set::default();
        for (idx, &update) in order.iter().enumerate() {
            // The update: a store of `acc + prod` back into a register slot.
            let &Op::Store { dst: acc, dst_idx, src, .. } = self.at(update) else {
                continue;
            };
            if !matches!(self.at(acc), Op::Storage { scope: MemScope::Register, .. }) || seen.contains(&acc) {
                continue;
            }
            let &Op::Binary { x, y, bop: BOp::Add } = self.at(src) else {
                continue;
            };
            // One side is the accumulator read at the stored slot: the same
            // index op, or an equal constant.
            let x_is_acc = matches!(self.at(x), Op::Load { src: s, src_idx } if *s == acc
                && (*src_idx == dst_idx
                    || self.resolve_const(*src_idx).and_then(Constant::as_dim)
                        == self.resolve_const(dst_idx).and_then(Constant::as_dim)));
            let y_is_acc = matches!(self.at(y), Op::Load { src: s, src_idx } if *s == acc
                && (*src_idx == dst_idx
                    || self.resolve_const(*src_idx).and_then(Constant::as_dim)
                        == self.resolve_const(dst_idx).and_then(Constant::as_dim)));
            if x_is_acc == y_is_acc {
                continue;
            }
            let mut prod = if x_is_acc { y } else { x };
            if let Op::Cast { x, .. } = self.at(prod) {
                prod = *x;
            }
            let &Op::Binary { x: la, y: lb, bop: BOp::Mul } = self.at(prod) else {
                continue;
            };
            let (&Op::Load { src: a_src, src_idx: a_idx }, &Op::Load { src: b_src, src_idx: b_idx }) = (self.at(la), self.at(lb))
            else {
                continue;
            };

            // The loop around the update: walk back to the innermost one
            // still open.
            let mut depth = 0usize;
            let mut k_loop = OpId::NULL;
            for &id in order[..idx].iter().rev() {
                match self.at(id) {
                    Op::EndLoop => depth += 1,
                    Op::Loop { .. } => {
                        if depth == 0 {
                            k_loop = id;
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
            }
            if k_loop.is_null() {
                continue;
            }
            let Some(loop_pos) = order[..idx].iter().rposition(|&id| id == k_loop) else {
                continue;
            };
            // Its matching EndLoop, then the store of the sum after it.
            let mut depth = 0usize;
            let mut end_pos = 0usize;
            for (pos, &id) in order.iter().enumerate().skip(loop_pos) {
                match self.at(id) {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => {
                        depth -= 1;
                        if depth == 0 {
                            end_pos = pos;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if end_pos == 0 {
                continue;
            }

            // Accumulator init before the loop.
            let Some(init_store) = order[..idx].iter().rev().copied().find(|&id| {
                matches!(self.at(id), Op::Store { dst, dst_idx: i, src, .. }
                    if *dst == acc
                        && (*i == dst_idx
                            || self.resolve_const(*i).and_then(Constant::as_dim)
                                == self.resolve_const(dst_idx).and_then(Constant::as_dim))
                        && matches!(self.at(*src), Op::Const(_)))
            }) else {
                continue;
            };

            // Final store of the reduced value after the loop (the value may
            // be cast on the way out, e.g. f32 accumulator stored as f16).
            let mut final_store = OpId::NULL;
            for &id in &order[end_pos + 1..] {
                let Op::Store { src, .. } = self.at(id) else {
                    continue;
                };
                let mut val = *src;
                while let Op::Cast { x, .. } | Op::Bitcast { x, .. } = self.at(val) {
                    val = *x;
                }
                let Op::Load { src: s, src_idx } = self.at(val) else {
                    continue;
                };
                if *s == acc
                    && (*src_idx == dst_idx
                        || self.resolve_const(*src_idx).and_then(Constant::as_dim)
                            == self.resolve_const(dst_idx).and_then(Constant::as_dim))
                {
                    final_store = id;
                    break;
                }
            }
            if final_store.is_null() {
                continue;
            }
            let &Op::Store { dst: c_src, dst_idx: c_idx, .. } = self.at(final_store) else {
                unreachable!()
            };

            // Affine views, oriented by the contraction: exactly one operand
            // has unit k stride, and that one is A (M x K); the other is B
            // (K x N).
            let (Some(map_la), Some(map_lb), Some(c_map)) =
                (self.get_strides(a_idx), self.get_strides(b_idx), self.get_strides(c_idx))
            else {
                continue;
            };
            // Orient by the OUTPUT map, not by which operand happens to have
            // unit k stride. C names the two output ranges {m,n}; the operand
            // view that indexes M is A, the one that indexes N is B, and both
            // contract over `k`. This is what admits the NT layout that
            // `x @ W.t()` produces, where A and B are *both* K-contiguous; the
            // old "exactly one operand has unit k stride" rule rejected it
            // before any fragment was built.
            let mut c_keys = c_map.iter().filter(|(v, _)| !v.is_null());
            let Some((&cr0, &(_, cs0))) = c_keys.next() else { continue };
            let Some((&cr1, &(_, cs1))) = c_keys.next() else { continue };
            // Row-major C: the row (M) range carries the larger stride.
            let (m_range, n_range) = if cs0 >= cs1 { (cr0, cr1) } else { (cr1, cr0) };
            if m_range == n_range {
                continue;
            }
            let (a_map, b_map, a_src, b_src) = if map_la.contains_key(&m_range) && map_lb.contains_key(&n_range) {
                (map_la, map_lb, a_src, b_src)
            } else if map_la.contains_key(&n_range) && map_lb.contains_key(&m_range) {
                (map_lb, map_la, b_src, a_src)
            } else {
                continue;
            };

            // Operand and accumulator dtypes, as the views are oriented.
            let a_dtype = match self.at(a_src) {
                Op::Param { dtype, .. } => *dtype,
                Op::Storage { dtype, .. } => *dtype,
                _ => continue,
            };
            let b_dtype = match self.at(b_src) {
                Op::Param { dtype, .. } => *dtype,
                Op::Storage { dtype, .. } => *dtype,
                _ => continue,
            };
            let c_dtype = match self.at(acc) {
                Op::Storage { dtype, .. } => *dtype,
                _ => continue,
            };

            // Both views name exactly {m,k} / {k,n}; C names {m,n}.
            if a_map.iter().filter(|(v, _)| !v.is_null()).count() != 2
                || b_map.iter().filter(|(v, _)| !v.is_null()).count() != 2
                || c_map.iter().filter(|(v, _)| !v.is_null()).count() != 2
                || !a_map.contains_key(&k_loop)
                || !b_map.contains_key(&k_loop)
                || !c_map.contains_key(&m_range)
                || !c_map.contains_key(&n_range)
            {
                continue;
            }
            if !matches!(self.at(m_range), Op::Range { kind: RangeKind::Group(_), .. })
                || !matches!(self.at(n_range), Op::Range { kind: RangeKind::Group(_), .. })
            {
                continue;
            }

            // Extents of the whole matmul.
            let m = a_map[&m_range].0;
            let k = a_map[&k_loop].0;
            let n = b_map[&n_range].0;

            // The whole span from the init store to the final store must be
            // this matmul: `apply` replaces it as one contiguous region.
            let Some(init_pos) = order.iter().position(|&id| id == init_store) else {
                continue;
            };
            let Some(final_pos) = order.iter().position(|&id| id == final_store) else {
                continue;
            };
            if init_pos >= loop_pos || final_pos <= end_pos {
                continue;
            }
            let mut clean = true;
            for &id in &order[init_pos..=final_pos] {
                match self.at(id) {
                    Op::Store { .. } if id != init_store && id != update && id != final_store => clean = false,
                    Op::Barrier => clean = false,
                    Op::Copy { .. } => clean = false,
                    Op::TT(_) => clean = false,
                    _ => {}
                }
            }
            if !clean {
                continue;
            }

            seen.insert(acc);

            // One config per device instruction this region fits: matching
            // operand/accumulator dtypes, and extents that are whole tiles
            // of that instruction (no padding yet).
            for variant in &info.mma_variants {
                if variant.a_dtype != a_dtype || variant.b_dtype != b_dtype || variant.c_dtype != c_dtype {
                    continue;
                }
                let (fm, fn_, fk) = {
                    let (fm, fn_, fk) = variant.dims.decompose_mnk();
                    (fm as Dim, fn_ as Dim, fk as Dim)
                };
                if m % fm != 0 || n % fn_ != 0 || k % fk != 0 {
                    continue;
                }
                regions.push(Region {
                    variant: variant.clone(),
                    init_store,
                    final_store,
                    a_src,
                    a_m: a_map[&m_range].1,
                    a_k: a_map[&k_loop].1,
                    b_src,
                    b_n: b_map[&n_range].1,
                    b_k: b_map[&k_loop].1,
                    c_src,
                    c_m: c_map[&m_range].1,
                    c_n: c_map[&n_range].1,
                    m_range,
                    n_range,
                    m,
                    n,
                    k,
                });
            }
        }
        Box::new(Tensorize { regions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        backend::{DTypeCapability, DeviceInfo, FragField},
        kernel::{Dev, MMADType, MMADims, split},
    };
    use std::sync::Arc;

    fn cuda() -> DeviceInfo {
        DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: true,
            warp_size: 32,
            cc: [7, 5],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            // Mirrors the CUDA `m16n8k8` row of `backend::cuda`'s table.
            mma_variants: vec![MMAVariant {
                dims: MMADims::m16n8k8,
                layout: crate::kernel::MMALayout::row_col,
                dtype: MMADType::f16_f16_f16_f32,
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                c_dtype: DType::F32,
                a: FragMap {
                    fields: vec![
                        FragField { axis: FragAxis::Row, shift: 2, width: 3, stride: 1 },
                        FragField { axis: FragAxis::Col, shift: 0, width: 2, stride: 2 },
                    ],
                    slots: vec![(0, 0), (0, 1), (8, 0), (8, 1)],
                },
                b: FragMap {
                    fields: vec![
                        FragField { axis: FragAxis::Row, shift: 2, width: 3, stride: 1 },
                        FragField { axis: FragAxis::Col, shift: 0, width: 2, stride: 2 },
                    ],
                    slots: vec![(0, 0), (0, 1)],
                },
                c: FragMap {
                    fields: vec![
                        FragField { axis: FragAxis::Row, shift: 2, width: 3, stride: 1 },
                        FragField { axis: FragAxis::Col, shift: 0, width: 2, stride: 2 },
                    ],
                    slots: vec![(0, 0), (0, 1), (8, 0), (8, 1)],
                },
            }],
            num_circular_buffers: 0,
            has_openmp: false,
        }
    }

    /// The scalar form (`acc = a[m,k]*b[k,n] + acc` in a K loop, then store)
    /// matches one region and becomes one `m16n8k8` update.
    #[test]
    fn matches_scalar_matmul_loop() {
        const M: i64 = 16;
        const N: i64 = 8;
        const K: i64 = 8;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(lv, N, n);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);

        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "one scalar matmul region");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 1, "one wmma update");
    }

    /// NT is the layout `x @ W.t()` produces: B is `[n,k]`, so *both* operands
    /// are K-contiguous. The matcher must orient by the output ranges instead
    /// of by which operand has unit k stride, or no tensorized candidate is
    /// generated for any llama projection.
    #[test]
    fn matches_scalar_matmul_loop_nt() {
        const M: i64 = 16;
        const N: i64 = 8;
        const K: i64 = 8;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            // B is [n, k]: k is the fast axis here too.
            let b_idx = k.mad(n, K, lv);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);

        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "one NT scalar matmul region");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 1, "one wmma update");
    }

    /// The exact seed IR a llama prefill projection linearizes to, transcribed
    /// from a `ZYX_DEBUG=264` dump (`llama matmul q_proj [128x3072]x[3072x3072]`):
    ///
    /// ```text
    /// param f16 a  (activation [m,k])   param f16 b  (weight [n,k])
    /// param mut f16 c16                 param mut f32 c32
    /// g0 = group_range(0) over 128      g1 = group_range(1) over 3072
    /// store acc = 0
    /// for k in 0..3072:
    ///     a = load a[g0*3072 + k]     b = load b[g1*3072 + k]
    ///     acc = f32(a*b) + load acc
    /// store c16[g0*3072 + g1] = f16(acc)
    /// store c32[g0*3072 + g1] = f32(f16(acc))
    /// ```
    ///
    /// Both operands are K-contiguous (NT: `x @ W.t()`), and there are **two**
    /// final stores — the f16 graph output and an f32 copy — which no other
    /// matcher test has.
    #[test]
    fn matches_llama_prefill_seed_ir() {
        const M: i64 = 128;
        const N: i64 = 3072;
        const K: i64 = 3072;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c16 = k.param_mut(DType::F16);
        let c32 = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(n, K, lv);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let res16 = k.cast(res, DType::F16);
        let res32 = k.cast(res16, DType::F32);
        let c_idx = k.mad(m, N, n);
        k.store(c16, res16, c_idx);
        k.store(c32, res32, c_idx);

        // The real seed reaches the search already linearized; match it in the
        // same form the autotuner sees.
        k.linearize();
        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "the llama prefill seed must select one region");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 1, "one wmma update");
    }

    /// Peephole target check: what does `opt_local_staging` see on an already
    /// tensorized kernel? Run both by hand and dump the IR.
    #[test]
    fn staging_on_tensorized_matmul() {
        const M: i64 = 16;
        const N: i64 = 8;
        const K: i64 = 64;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(n, K, lv);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);
        k.linearize();

        let t = k.opt_tensorize();
        println!("tensorize nconfigs = {}", t.nconfigs());
        assert_eq!(t.nconfigs(), 1, "tensorize must match the seed");
        t.apply(&mut k, 0);
        k.verify();
        println!("--- after tensorize ---");
        k.debug();

        let s = k.opt_local_staging();
        println!("staging nconfigs = {}", s.nconfigs());
        if s.nconfigs() > 0 {
            s.apply(&mut k, 0);
            k.verify();
            println!("--- after staging ---");
            k.debug();
        }
    }

    /// The fragment-slot pattern, matched against a tensorized kernel's A
    /// slots. Bindings: `l` lane range, `mb` tile row base (`m*16`), `kb`
    /// k base (`k*8`), `K` the row stride, `s` the slot offset.
    #[test]
    #[ignore = "the epilogue-simplified slot index is context-dependent; the structural FragMap matcher replaces this"]
    fn mma_fragment_slot_pattern_matches() {
        use crate::kernel::pat::Pat;

        const M: i64 = 16;
        const N: i64 = 8;
        const K: i64 = 64;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(n, K, lv);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);
        k.linearize();
        k.opt_tensorize().apply(&mut k, 0);
        k.default_epilogue();
        k.verify();
        println!("--- after epilogue ---");
        k.debug();

        // `index = (mb + lane/4 % 8) * K + (kb + lane%4 * 2 + s)`
        let l = Pat::bind('l');
        let row = (l.clone() / 4) % 8;
        let col = (l.clone() % 4) * 2;
        let idx = (Pat::bind('m') + row) * Pat::bind_const('K')
            + (Pat::bind('k') + col + Pat::bind_const('s'));
        let slot = Pat::load(Pat::bind('a'), idx);

        let wmma = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Wmma { .. }).then_some(id))
            .expect("tensorize emitted a wmma");
        let Op::Wmma { a: a_frag, .. } = *k.at(wmma) else { unreachable!() };
        let Op::Stack { ref ops } = *k.at(a_frag) else { panic!("A operand is not a stack") };
        let mut matched = 0;
        for &slot_op in ops.iter() {
            if let Some(b) = k.match_pat(slot_op, &slot) {
                matched += 1;
                println!("slot {slot_op}: s={} K={}", b.int('s'), b.int('K'));
            }
        }
        assert_eq!(matched, ops.len(), "every A slot must match the fragment pattern");
    }

    /// Staging on a tensorized kernel: the MMA staging pass must stage both
    /// fragment operands, and after it the wmma reads shared memory.
    #[test]
    fn mma_staging_rewrites_fragments() {
        const M: i64 = 16;
        const N: i64 = 8;
        const K: i64 = 64;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(n, K, lv);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);
        k.linearize();
        k.opt_tensorize().apply(&mut k, 0);

        let staging = k.opt_mma_staging();
        assert_eq!(staging.nconfigs(), 2, "A and B are staged");
        staging.apply(&mut k, 0);
        staging.apply(&mut k, 1);
        k.default_epilogue();
        k.verify();

        // The wmma now reads a fragment whose slots load shared storage.
        let wmma = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Wmma { .. }).then_some(id))
            .expect("wmma present");
        let Op::Wmma { a: a_frag, b: b_frag, .. } = *k.at(wmma) else { unreachable!() };
        for operand in [a_frag, b_frag] {
            let Op::Stack { ref ops } = *k.at(operand) else { panic!("operand is not a stack") };
            for &slot in ops.iter() {
                let Op::Load { src, .. } = *k.at(slot) else { panic!("slot is not a load") };
                assert!(
                    matches!(k.at(src), Op::Storage { scope: MemScope::Local, .. }),
                    "slot must read the staged shared tile"
                );
            }
        }
    }

    /// Tiling is the split's job, not the pass's: splitting the M and N tile
    /// ranges into unroll lanes multiplies the tile per warp to 4x8 fragments,
    /// and `devectorize` duplicates the mma per lane. The pass itself stays
    /// one fragment; the accumulator grows to 32 fragments x 4 slots.
    #[test]
    fn split_duplicates_the_wmma() {
        const M: i64 = 128;
        const N: i64 = 128;
        const K: i64 = 32;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, lv| {
            let a_idx = k.mad(m, K, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(lv, N, n);
            let b_load = k.load(b, b_idx);
            let prod = k.mul(a_load, b_load);
            let prod = k.cast(prod, DType::F32);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let res = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, res, c_idx);

        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "one scalar matmul region");
        opt.apply(&mut k, 0);
        k.verify();

        // The tile ranges the pass left behind: M on group axis 0, N on 1.
        let mut ranges: Vec<(u32, OpId)> = Vec::new();
        let mut op_id = k.head;
        while !op_id.is_null() {
            if let Op::Range { axis, kind: RangeKind::Group(_) } = k.at(op_id) {
                ranges.push((*axis, op_id));
            }
            op_id = k.next_op(op_id);
        }
        let m_range = *ranges.iter().find(|(axis, _)| *axis == 0).map(|(_, id)| id).expect("M tile range");
        let n_range = *ranges.iter().find(|(axis, _)| *axis == 1).map(|(_, id)| id).expect("N tile range");

        let m_split = split::OptSplit { configs: vec![(m_range, split::SplitMode::Unroll, 4)] };
        Optimization::apply(&m_split, &mut k, 0);
        let n_split = split::OptSplit { configs: vec![(n_range, split::SplitMode::Unroll, 8)] };
        Optimization::apply(&n_split, &mut k, 0);
        k.devectorize();

        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 32, "4 M fragments x 8 N fragments, one mma each");
        let no_unroll = !k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. }));
        assert!(no_unroll, "devectorize unfolds the split lanes");
        let slots = k
            .iter_unordered()
            .filter(|(_, op)| matches!(op, Op::Storage { scope: MemScope::Register, dtype: DType::F32, .. }))
            .map(|(_, op)| match op {
                Op::Storage { len, .. } => *len,
                _ => unreachable!("filtered to register storages"),
            })
            .max()
            .expect("the accumulator storage");
        assert_eq!(slots, 128, "32 fragments x 4 slots");
    }
}
