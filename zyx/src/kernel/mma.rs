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
    Set, Map,
    backend::{FragAxis, FragMap, MMAVariant},
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, autotune::Optimization},
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

/// One output store of a tiled region and the cast chain from the
/// accumulator to it.
#[derive(Debug, Clone)]
struct Epi {
    store: OpId,
    dst: OpId,
    /// Cast dtypes, outermost first.
    chain: Vec<crate::DType>,
    /// Non-lane index terms and the constant offset.
    fixed: Vec<(OpId, Dim)>,
    offset: Dim,
    /// Output strides of the m and n lane axes.
    m: Dim,
    n: Dim,
}

/// A matmul region already tiled by `split`: the update is
/// `acc + reduce sum(A * B) over the k unroll lane`, and the three lane axes
/// carry the tile grid.
#[derive(Debug, Clone)]
struct Tiled {
    variant: MMAVariant,
    /// Accumulator storage, indexed by the m/n unroll lanes.
    acc: OpId,
    a_src: OpId,
    b_src: OpId,
    /// Index terms of each view with the lane axes removed.
    a_fixed: Vec<(OpId, Dim)>,
    a_offset: Dim,
    a_m: Dim,
    a_k: Dim,
    b_fixed: Vec<(OpId, Dim)>,
    b_offset: Dim,
    b_n: Dim,
    b_k: Dim,
    /// Lane extents (tile grid dimensions in instruction tiles).
    mt: Dim,
    nt: Dim,
    kt: Dim,
    /// Unroll extents of the m and n axes (the accumulator's row width).
    m_len: Dim,
    n_len: Dim,
    /// Fragment slots per tile (the new accumulator length is
    /// `mt * nt * slots`).
    slots: Dim,
    /// The accumulator's zero initialisation.
    init: OpId,
    /// The loop the update runs in.
    loop_id: OpId,
    /// The update store the reduce feeds.
    update: OpId,
    epilogue: Vec<Epi>,
}

/// The tensorization optimization: one config per validated region.
#[derive(Debug)]
struct Tensorize {
    regions: Vec<Region>,
    tiled: Vec<Tiled>,
}

impl Optimization for Tensorize {
    fn nconfigs(&self) -> u64 {
        (self.regions.len() + self.tiled.len()) as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        let config = config as usize;
        if config >= self.regions.len() {
            if let Some(t) = self.tiled.get(config - self.regions.len()) {
                tile_to_wmma(kernel, t);
            }
            return;
        }
        let Some(r) = self.regions.get(config) else {
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

/// Sum of the non-lane index terms plus a constant offset.
fn base_expr(kernel: &mut Kernel, before: OpId, fixed: &[(OpId, Dim)], offset: Dim, extra: Dim) -> OpId {
    let mut base = kernel.insert_before(before, Op::Const(Constant::idx(offset + extra)));
    for &(var, stride) in fixed {
        let s = kernel.insert_before(before, Op::Const(Constant::idx(stride)));
        let term = kernel.insert_before(before, Op::Binary { x: s, y: var, bop: BOp::Mul });
        base = kernel.insert_before(before, Op::Binary { x: base, y: term, bop: BOp::Add });
    }
    base
}

/// The lane's coordinate along one fragment axis, from the bit-fields.
fn lane_coord(kernel: &mut Kernel, before: OpId, lane: OpId, map: &FragMap, axis: FragAxis, warp: u32) -> OpId {
    let mut sum = kernel.insert_before(before, Op::Const(Constant::idx(0)));
    for f in map.fields.iter().filter(|f| f.axis == axis) {
        let mut term = lane;
        if f.shift > 0 {
            let d = kernel.insert_before(before, Op::Const(Constant::idx(Dim::from(1i64 << f.shift))));
            term = kernel.insert_before(before, Op::Binary { x: term, y: d, bop: BOp::Div });
        }
        if (1u32 << f.width) < warp {
            let w = kernel.insert_before(before, Op::Const(Constant::idx(Dim::from(1i64 << f.width))));
            term = kernel.insert_before(before, Op::Binary { x: term, y: w, bop: BOp::Mod });
        }
        if f.stride != 1 {
            let s = kernel.insert_before(before, Op::Const(Constant::idx(Dim::from(f.stride))));
            term = kernel.insert_before(before, Op::Binary { x: term, y: s, bop: BOp::Mul });
        }
        sum = kernel.insert_before(before, Op::Binary { x: sum, y: term, bop: BOp::Add });
    }
    sum
}

/// One fragment slot's element address: `base + row * sr + col * sc + slot`.
fn slot_addr(kernel: &mut Kernel, before: OpId, lane: OpId, map: &FragMap, base: OpId, sr: Dim, sc: Dim, slot: (i16, i16), warp: u32) -> OpId {
    let mut addr = base;
    for axis in [FragAxis::Row, FragAxis::Col] {
        let stride = if axis == FragAxis::Row { sr } else { sc };
        if stride == 0 {
            continue;
        }
        let term = lane_coord(kernel, before, lane, map, axis, warp);
        let term = if stride == 1 {
            term
        } else {
            let s = kernel.insert_before(before, Op::Const(Constant::idx(stride)));
            kernel.insert_before(before, Op::Binary { x: term, y: s, bop: BOp::Mul })
        };
        addr = kernel.insert_before(before, Op::Binary { x: addr, y: term, bop: BOp::Add });
    }
    let off = Dim::from(slot.0) * sr + Dim::from(slot.1) * sc;
    if off != 0 {
        let c = kernel.insert_before(before, Op::Const(Constant::idx(off)));
        addr = kernel.insert_before(before, Op::Binary { x: addr, y: c, bop: BOp::Add });
    }
    addr
}

/// Element index into the tiled accumulator for a fragment slot of tile
/// `(tm, tn)`: `(tm * nt + tn) * slots + s`.
fn acc_index(_kernel: &mut Kernel, _before: OpId, _lane: OpId, _map: &FragMap, t: &Tiled, _fm: Dim, _fn: Dim, tm: Dim, tn: Dim, slot: usize, _warp: u32) -> OpId {
    let idx = (tm * t.nt + tn) * t.slots + Dim::from(slot as i64);
    _kernel.insert_before(_before, Op::Const(Constant::idx(idx)))
}

/// A fragment operand's slot values, pairing slots that are contiguous in
/// memory into one vector load (the `half2` the instruction consumes).
#[allow(clippy::too_many_arguments)]
fn frag_elems(kernel: &mut Kernel, before: OpId, lane: OpId, map: &FragMap, src: OpId, base: OpId, sr: Dim, sc: Dim, warp: u32) -> Vec<OpId> {
    let slots = &map.slots;
    let mut out = Vec::with_capacity(slots.len());
    let mut i = 0;
    while i < slots.len() {
        if i + 1 < slots.len() {
            let a = slots[i];
            let b = slots[i + 1];
            let delta = Dim::from(b.0 - a.0) * sr + Dim::from(b.1 - a.1) * sc;
            if delta == 1 {
                let addr = slot_addr(kernel, before, lane, map, base, sr, sc, a, warp);
                let un = kernel.insert_before(before, Op::Unroll { axis: 0, len: 2 });
                let idx = kernel.insert_before(before, Op::Binary { x: addr, y: un, bop: BOp::Add });
                let v = kernel.insert_before(before, Op::Load { src, src_idx: idx });
                out.push(kernel.insert_before(before, Op::Index { vec: v, idx: 0 }));
                out.push(kernel.insert_before(before, Op::Index { vec: v, idx: 1 }));
                i += 2;
                continue;
            }
        }
        let addr = slot_addr(kernel, before, lane, map, base, sr, sc, slots[i], warp);
        out.push(kernel.insert_before(before, Op::Load { src, src_idx: addr }));
        i += 1;
    }
    out
}

/// Emit a matched tiled region as wmma instructions, in place: the
/// accumulator storage keeps its identity, its accesses are rewritten in
/// fragment order, and the split's reduce/update and the output stores are
/// replaced (DCE drops their cones).
fn tile_to_wmma(kernel: &mut Kernel, t: &Tiled) {
    let v = &t.variant;
    let (fm, fn_, fk) = {
        let (m, n, k) = v.dims.decompose_mnk();
        (m as Dim, n as Dim, k as Dim)
    };
    let warp = u32::from(kernel.device_info().warp_size);
    let lane = kernel.insert_before(t.loop_id, Op::Range { axis: 0, kind: RangeKind::Local(warp) });

    // The accumulator holds one register per (tile, slot): `m_len * n_len`
    // scalar elements would be 32x what the instruction layout needs.
    let len = t.mt * t.nt * t.slots;
    if let Op::Storage { len: l, .. } = &mut kernel.ops[t.acc].op {
        *l = len;
    }
    // Zero it: the scalar init store wrote the whole old tile.
    let zero = match kernel.at(t.init) {
        Op::Store { src, .. } => *src,
        _ => unreachable!("tiled tensorize: init is not a store"),
    };
    for s in 0..len {
        let idx = kernel.insert_before(t.init, Op::Const(Constant::idx(s)));
        kernel.insert_before(t.init, Op::Store { dst: t.acc, dst_idx: idx, src: zero });
    }
    kernel.remove_op(t.init);

    for tm in 0..t.mt {
        for tn in 0..t.nt {
            for tk in 0..t.kt {
                let a_base =
                    base_expr(kernel, t.update, &t.a_fixed, t.a_offset, tm * fm * t.a_m + tk * fk * t.a_k);
                let a_elems = frag_elems(kernel, t.update, lane, &v.a, t.a_src, a_base, t.a_m, t.a_k, warp);
                let a_frag = kernel.insert_before(t.update, Op::Stack { ops: a_elems.into() });
                let b_base =
                    base_expr(kernel, t.update, &t.b_fixed, t.b_offset, tn * fn_ * t.b_n + tk * fk * t.b_k);
                let b_elems = frag_elems(kernel, t.update, lane, &v.b, t.b_src, b_base, t.b_n, t.b_k, warp);
                let b_frag = kernel.insert_before(t.update, Op::Stack { ops: b_elems.into() });
                let mut c_elems = Vec::with_capacity(v.c.slots.len());
                let mut c_idx = Vec::with_capacity(v.c.slots.len());
                for (s, _slot) in v.c.slots.iter().enumerate() {
                    let idx = acc_index(kernel, t.update, lane, &v.c, t, fm, fn_, tm, tn, s, warp);
                    c_idx.push(idx);
                    c_elems.push(kernel.insert_before(t.update, Op::Load { src: t.acc, src_idx: idx }));
                }
                let c_frag = kernel.insert_before(t.update, Op::Stack { ops: c_elems.into() });
                let d = kernel.insert_before(
                    t.update,
                    Op::Wmma { dims: v.dims, layout: v.layout, dtype: v.dtype, a: a_frag, b: b_frag, c: c_frag },
                );
                for (i, &idx) in c_idx.iter().enumerate() {
                    let elem = kernel.insert_before(t.update, Op::Index { vec: d, idx: i });
                    kernel.insert_before(t.update, Op::Store { dst: t.acc, dst_idx: idx, src: elem });
                }
            }
        }
    }

    // Output stores: the accumulator keeps the values, each slot is written
    // at its own output coordinate.
    for epi in &t.epilogue {
        for tm in 0..t.mt {
            for tn in 0..t.nt {
                for s in 0..v.c.slots.len() {
                    let slot = v.c.slots[s];
                    let idx = acc_index(kernel, epi.store, lane, &v.c, t, fm, fn_, tm, tn, s, warp);
                    let mut elem = kernel.insert_before(epi.store, Op::Load { src: t.acc, src_idx: idx });
                    for dtype in epi.chain.iter().rev() {
                        elem = kernel.insert_before(epi.store, Op::Cast { x: elem, dtype: *dtype });
                    }
                    let extra = (tm * fm + Dim::from(slot.0)) * epi.m + (tn * fn_ + Dim::from(slot.1)) * epi.n;                    let mut addr = base_expr(kernel, epi.store, &epi.fixed, epi.offset, extra);
                    let row = lane_coord(kernel, epi.store, lane, &v.c, FragAxis::Row, warp);
                    if epi.m != 1 {
                        let s = kernel.insert_before(epi.store, Op::Const(Constant::idx(epi.m)));
                        let t2 = kernel.insert_before(epi.store, Op::Binary { x: row, y: s, bop: BOp::Mul });
                        addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: t2, bop: BOp::Add });
                    } else {
                        addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: row, bop: BOp::Add });
                    }
                    let col = lane_coord(kernel, epi.store, lane, &v.c, FragAxis::Col, warp);
                    if epi.n != 1 {
                        let s = kernel.insert_before(epi.store, Op::Const(Constant::idx(epi.n)));
                        let t2 = kernel.insert_before(epi.store, Op::Binary { x: col, y: s, bop: BOp::Mul });
                        addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: t2, bop: BOp::Add });
                    } else {
                        addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: col, bop: BOp::Add });
                    }
                    kernel.insert_before(epi.store, Op::Store { dst: epi.dst, dst_idx: addr, src: elem });
                }
            }
        }
    }

    let mut doomed: Vec<OpId> = vec![t.update];
    doomed.extend(t.epilogue.iter().map(|e| e.store));
    for id in doomed {
        kernel.remove_op(id);
    }
    kernel.verify();
}

impl Kernel {
    /// Make the tensorization optimization: scan the linearized IR for
    /// scalar matmul regions, one config per region the device can
    /// accelerate. Never mutates.
    pub fn opt_tensorize(&self) -> Box<dyn Optimization> {
        self.tensorize_impl(false)
    }

    /// The tiled-form configs only: used by the tensor recipe, which has
    /// already done the splits. Never mutates.
    pub(crate) fn opt_tensorize_tiled(&self) -> Box<dyn Optimization> {
        self.tensorize_impl(true)
    }

    fn tensorize_impl(&self, tiled_only: bool) -> Box<dyn Optimization> {
        let mut regions = Vec::new();
        let info = self.device_info();
        {
            // Device gate: a tensor-core unit offering at least one
            // instruction variant.
            if !info.tensor_cores || info.mma_variants.is_empty() {
                return Box::new(Tensorize { regions, tiled: Vec::new() });
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
        // ---- Tiled form: `split` has already built the tile grid, and the
        // update is `acc + reduce sum(A * B) over the k lane`. Match the
        // reduce, take the m/n lane axes from the two index maps, and let
        // `apply` emit the wmmas in place.
        let mut tiled: Vec<Tiled> = Vec::new();
        let has_unroll = self.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. }));
        let has_local = self.iter_unordered().any(|(_, op)| matches!(op, Op::Range { kind: RangeKind::Local(_), .. }));
        if has_unroll && !has_local {
            let mut order: Vec<OpId> = Vec::new();
            let mut id = self.head;
            while !id.is_null() {
                order.push(id);
                id = self.next_op(id);
            }
            for (red, op) in self.iter_unordered() {
                let Op::Reduce { x: red_src, rop: BOp::Add, .. } = *op else {
                    continue;
                };
                let mut v = red_src;
                while let Op::Cast { x, .. } | Op::Bitcast { x, .. } = *self.at(v) {
                    v = x;
                }
                let Op::Binary { x: la, y: lb, bop: BOp::Mul } = *self.at(v) else {
                    continue;
                };
                let (Op::Load { src: a_src, src_idx: a_idx }, Op::Load { src: b_src, src_idx: b_idx }) =
                    (self.at(la), self.at(lb))
                else {
                    continue;
                };
                let (a_src, a_idx, b_src, b_idx) = (*a_src, *a_idx, *b_src, *b_idx);
                if !matches!(self.at(a_src), Op::Param { kind: ParamKind::Global, .. })
                    || !matches!(self.at(b_src), Op::Param { kind: ParamKind::Global, .. })
                {
                    continue;
                }
                let (Some(a_map), Some(b_map)) = (self.get_strides(a_idx), self.get_strides(b_idx)) else {
                    continue;
                };
                // The contraction lane is the unroll axis both operands share;
                // m and n are the remaining unroll axis of each.
                let unroll_keys = |map: &Map<OpId, (Dim, Dim)>| -> Vec<(OpId, u16)> {
                    map.keys()
                        .filter(|&&var| !var.is_null())
                        .filter_map(|&var| match *self.at(var) {
                            Op::Unroll { len, .. } => Some((var, len)),
                            _ => None,
                        })
                        .collect()
                };
                let a_unrolls = unroll_keys(&a_map);
                let b_unrolls = unroll_keys(&b_map);
                let Some(&(k_var, k_len_u16)) = a_unrolls.iter().find(|(v, _)| b_unrolls.iter().any(|(w, _)| w == v))
                else {
                    continue;
                };
                if a_unrolls.iter().filter(|(v, _)| *v != k_var).count() != 1
                    || b_unrolls.iter().filter(|(v, _)| *v != k_var).count() != 1
                {
                    continue;
                }
                let (m_var, m_len_u16) = *a_unrolls.iter().find(|(v, _)| *v != k_var).unwrap();
                let (n_var, n_len_u16) = *b_unrolls.iter().find(|(v, _)| *v != k_var).unwrap();
                let k_len = Dim::from(k_len_u16);
                if m_var == n_var {
                    continue;
                }
                let (m_len, n_len) = (Dim::from(m_len_u16), Dim::from(n_len_u16));

                // The update: `store acc = add(reduce, load acc)`.
                let mut update = OpId::NULL;
                let mut acc = OpId::NULL;
                for (sid, sop) in self.iter_unordered() {
                    let Op::Store { dst, src, .. } = *sop else {
                        continue;
                    };
                    let Op::Binary { x: ux, y: uy, bop: BOp::Add } = *self.at(src) else {
                        continue;
                    };
                    let other = if ux == red {
                        uy
                    } else if uy == red {
                        ux
                    } else {
                        continue;
                    };
                    let Op::Load { src: s, .. } = *self.at(other) else {
                        continue;
                    };
                    if s == dst && matches!(self.at(s), Op::Storage { scope: MemScope::Register, .. }) {
                        update = sid;
                        acc = s;
                        break;
                    }
                }
                if update.is_null() {
                    continue;
                }

                // The accumulator's zero initialisation.
                let mut init = OpId::NULL;
                for (sid, sop) in self.iter_unordered() {
                    if let Op::Store { dst, src, .. } = *sop
                        && dst == acc
                        && matches!(self.at(src), Op::Const(_))
                    {
                        init = sid;
                        break;
                    }
                }
                if init.is_null() {
                    continue;
                }

                // The loop the update runs in.
                let Some(red_pos) = order.iter().position(|&x| x == red) else {
                    continue;
                };
                let mut depth = 0usize;
                let mut loop_id = OpId::NULL;
                for &oid in order[..red_pos].iter().rev() {
                    match self.at(oid) {
                        Op::EndLoop => depth += 1,
                        Op::Loop { .. } => {
                            if depth == 0 {
                                loop_id = oid;
                                break;
                            }
                            depth -= 1;
                        }
                        _ => {}
                    }
                }
                if loop_id.is_null() {
                    continue;
                }

                // Output stores: every store whose value is the accumulator.
                let mut epilogue: Vec<Epi> = Vec::new();
                for (sid, sop) in self.iter_unordered() {
                    if sid == update {
                        continue;
                    }
                    let Op::Store { dst, src, dst_idx } = *sop else {
                        continue;
                    };
                    let mut chain = Vec::new();
                    let mut val = src;
                    let mut hit = false;
                    loop {
                        match self.at(val) {
                            Op::Cast { x, dtype } | Op::Bitcast { x, dtype } => {
                                chain.push(*dtype);
                                val = *x;
                            }
                            Op::Load { src: s, .. } if *s == acc => {
                                hit = true;
                                break;
                            }
                            _ => break,
                        }
                    }
                    if !hit {
                        continue;
                    }
                    let Some(cmap) = self.get_strides(dst_idx) else {
                        continue;
                    };
                    let (Some(&(_, c_m)), Some(&(_, c_n))) = (cmap.get(&m_var), cmap.get(&n_var)) else {
                        continue;
                    };
                    let mut fixed = Vec::new();
                    let mut offset = 0;
                    for (&var, &(_, stride)) in cmap.iter() {
                        if var.is_null() {
                            offset = stride;
                        } else if var != m_var && var != n_var {
                            fixed.push((var, stride));
                        }
                    }
                    epilogue.push(Epi { store: sid, dst, chain, fixed, offset, m: c_m, n: c_n });
                }

                // Index terms without the tile axes.
                let fixed_terms = |map: &Map<OpId, (Dim, Dim)>, a: OpId, b: OpId| {
                    let mut fixed = Vec::new();
                    let mut offset = 0;
                    for (&var, &(_, stride)) in map.iter() {
                        if var.is_null() {
                            offset = stride;
                        } else if var != a && var != b {
                            fixed.push((var, stride));
                        }
                    }
                    (fixed, offset)
                };
                let (a_fixed, a_offset) = fixed_terms(&a_map, m_var, k_var);
                let (b_fixed, b_offset) = fixed_terms(&b_map, n_var, k_var);

                let a_dtype = match self.at(a_src) {
                    Op::Param { dtype, .. } => *dtype,
                    _ => continue,
                };
                let b_dtype = match self.at(b_src) {
                    Op::Param { dtype, .. } => *dtype,
                    _ => continue,
                };
                let c_dtype = match self.at(acc) {
                    Op::Storage { dtype, .. } => *dtype,
                    _ => continue,
                };
                for variant in &info.mma_variants {
                    if variant.a_dtype != a_dtype || variant.b_dtype != b_dtype || variant.c_dtype != c_dtype {
                        continue;
                    }
                    let (fm, fn_, fk) = {
                        let (fm, fn_, fk) = variant.dims.decompose_mnk();
                        (fm as Dim, fn_ as Dim, fk as Dim)
                    };
                    if m_len % fm != 0 || n_len % fn_ != 0 || k_len % fk != 0 {
                        continue;
                    }
                    let (mt, nt, kt) = (m_len / fm, n_len / fn_, k_len / fk);
                    if mt.saturating_mul(nt).saturating_mul(kt) > 256 {
                        continue;
                    }
                    tiled.push(Tiled {
                        variant: variant.clone(),
                        acc,
                        a_src,
                        b_src,
                        a_fixed: a_fixed.clone(),
                        a_offset,
                        a_m: a_map[&m_var].1,
                        a_k: a_map[&k_var].1,
                        b_fixed: b_fixed.clone(),
                        b_offset,
                        b_n: b_map[&n_var].1,
                        b_k: b_map[&k_var].1,
                        mt,
                        nt,
                        kt,
                        m_len,
                        n_len,
                        slots: Dim::from(variant.c.slots.len() as i64),
                        init,
                        loop_id,
                        update,
                        epilogue: epilogue.clone(),
                    });
                }
            }
        }
        if tiled_only {
            // The recipe only wants the tiled configs; the scalar scan above
            // ran against a kernel that no longer has a scalar region.
            regions.clear();
        }
        Box::new(Tensorize { regions, tiled })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType, Tensor,
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

        // Register block: `split` on the narrowed tile ranges, then the
        // epilogue's devectorize unfolds the lanes into one wmma per tile.
        k.split(m, split::SplitMode::Unroll, 4);
        k.split(n, split::SplitMode::Unroll, 4);
        k.default_epilogue();
        k.device_epilogue();
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 16, "4x4 register block");

        let compiled = match k.compile() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping the timed run: {e}");
                return;
            }
        };
        let dev = compiled.device();
        let a = Tensor::rand([M, K], DType::F16).unwrap().to(dev).unwrap();
        let b = Tensor::rand([N, K], DType::F16).unwrap().to(dev).unwrap();
        let shapes = || vec![[M, N], [M, N]];
        compiled.forward(&[&a, &b], shapes()).unwrap();
        dev.synchronize().unwrap();
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let out = compiled.forward(&[&a, &b], shapes()).unwrap();
            std::hint::black_box(&out);
        }
        dev.synchronize().unwrap();
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let flop = 2.0 * M as f64 * N as f64 * K as f64;
        println!("tensorize+split(4,4): {:.3} ms, {:.1} GFLOP/s", dt * 1e3, flop / dt / 1e9);
    }

    /// The tiled form: split first (no Local split — the accumulator must be
    /// the whole per-thread tile), then tensorize matches the reduce and
    /// emits the wmmas in place.
    #[test]
    fn tiled_split_tensorizes() {
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
        let mut k_loop = OpId::NULL;
        k.loop_over(K, |k, lv| {
            k_loop = lv;
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

        k.split(m, split::SplitMode::Unroll, 64);
        k.split(n, split::SplitMode::Unroll, 32);
        k.split(k_loop, split::SplitMode::Unroll, 16);
        k.verify();

        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "the tiled form matches one region");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 32, "4 m tiles x 4 n tiles x 2 k steps");
        k.dead_code_elimination();
        let reduces = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).count();
        assert_eq!(reduces, 0, "the reduce is replaced");

        // Speed of the tiled match on the prefill projection.
        k.default_epilogue();
        k.device_epilogue();
        let compiled = match k.compile() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping the timed run: {e}");
                return;
            }
        };
        let dev = compiled.device();
        let a = Tensor::rand([M, K], DType::F16).unwrap().to(dev).unwrap();
        let b = Tensor::rand([N, K], DType::F16).unwrap().to(dev).unwrap();
        let shapes = || vec![[M, N], [M, N]];
        compiled.forward(&[&a, &b], shapes()).unwrap();
        dev.synchronize().unwrap();
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let out = compiled.forward(&[&a, &b], shapes()).unwrap();
            std::hint::black_box(&out);
        }
        dev.synchronize().unwrap();
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let flop = 2.0 * M as f64 * N as f64 * K as f64;
        println!("tiled tensorize (64x32x16 lanes): {:.3} ms, {:.1} GFLOP/s", dt * 1e3, flop / dt / 1e9);
    }

    /// The stored IR must round-trip exactly: same ops, same device snapshot,
    /// same hash, no trailing bytes. Serialization is the bench database's
    /// source of truth, so anything lossy here is silent data corruption.
    #[test]
    fn kernel_ir_round_trips() {
        use nanoserde::{DeBin, SerBin};

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
        let mut k_loop = OpId::NULL;
        k.loop_over(K, |k, lv| {
            k_loop = lv;
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

        k.split(m, split::SplitMode::Unroll, 64);
        k.split(n, split::SplitMode::Unroll, 32);
        k.split(k_loop, split::SplitMode::Unroll, 16);
        k.opt_tensorize().apply(&mut k, 0);
        k.default_epilogue();
        k.verify();
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Stack { .. })),
            "the tensorized kernel has fragment stacks"
        );

        let bytes = k.serialize_bin();
        let mut offset = 0;
        let back = Kernel::de_bin(&mut offset, &bytes).expect("stored IR deserializes");
        assert_eq!(offset, bytes.len(), "deserialize left trailing bytes");
        assert_eq!(back, k, "round-tripped IR differs");
        assert_eq!(back.get_hash(), k.get_hash(), "round-trip changed the kernel identity");
        assert_eq!(back.dev_info(), k.dev_info(), "round-trip lost the device snapshot");
        back.verify();
    }

    /// The tiled tensorize on a small shape, checked against a CPU reference:
    /// the lane decomposition has to reproduce exactly the scalar matmul.
    #[test]
    fn tiled_split_values() {
        const M: i64 = 32;
        const N: i64 = 16;
        const K: i64 = 32;

        let build = || {
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
            let mut k_loop = OpId::NULL;
            k.loop_over(K, |k, lv| {
                k_loop = lv;
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
            (k, m, n, k_loop)
        };

        let run = |k: Kernel, a: &Tensor, b: &Tensor| -> Option<Vec<f32>> {
            let compiled = match k.compile() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("skipping: {e}");
                    return None;
                }
            };
            let out = compiled.forward(&[a, b], vec![[M, N], [M, N]]).unwrap();
            Some(out[1].clone().try_into().unwrap())
        };

        let (mut k, m, n, k_loop) = build();
        assert!(!k.split(m, split::SplitMode::Unroll, 32).is_null(), "m split");
        assert!(!k.split(n, split::SplitMode::Unroll, 16).is_null(), "n split");
        assert!(!k.split(k_loop, split::SplitMode::Unroll, 16).is_null(), "k split");
        k.verify();
        let scalar = k.clone();
        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 1, "the tiled form matches");
        opt.apply(&mut k, 0);
        k.default_epilogue();
        k.device_epilogue();
        k.verify();

        // One input pair for both kernels: resolve the device once.
        let dev = match Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) {
            Some(d) => d,
            None => {
                eprintln!("skipping: no device");
                return;
            }
        };
        let a = Tensor::rand([M, K], DType::F16).unwrap().to(dev).unwrap();
        let b = Tensor::rand([N, K], DType::F16).unwrap().to(dev).unwrap();
        let a_host: Vec<f32> = a.clone().cast(DType::F32).try_into().unwrap();
        let b_host: Vec<f32> = b.clone().cast(DType::F32).try_into().unwrap();

        let Some(got) = run(k, &a, &b) else { return };
        let Some(want) = run(scalar, &a, &b) else { return };
        let mut worst = 0.0f32;
        for i in 0..(M * N) as usize {
            worst = worst.max((got[i] - want[i]).abs());
        }
        println!("tiled vs scalar: max |err| = {worst}");
        assert!(worst < 1e-2, "tiled tensorize disagrees with the scalar kernel: {worst}");
        // And the scalar kernel itself matches a host dot product.
        let mut host_worst = 0.0f32;
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut sum = 0.0f32;
                for t in 0..K as usize {
                    sum += a_host[i * K as usize + t] * b_host[j * K as usize + t];
                }
                host_worst = host_worst.max((want[i * N as usize + j] - sum).abs());
            }
        }
        println!("scalar vs host: max |err| = {host_worst}");
        assert!(host_worst < 1e-2, "scalar split kernel is wrong: {host_worst}");
    }

    /// The recipe wires the splits to the tiled matcher: one move from the
    /// plain seed IR to the tensorized kernel.
    #[test]
    fn tensor_recipe_tensorizes() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
        let c16 = k.param_mut(DType::F16);
        let c32 = k.param_mut(DType::F32);
        let m = k.group_range(0, 128i64);
        let n = k.group_range(1, 3072i64);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(3072i64, |k, lv| {
            let a_idx = k.mad(m, 3072i64, lv);
            let a_load = k.load(a, a_idx);
            let b_idx = k.mad(n, 3072i64, lv);
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
        let c_idx = k.mad(m, 3072i64, n);
        k.store(c16, res16, c_idx);
        k.store(c32, res32, c_idx);

        let recipe = k.opt_tensor_recipe();
        assert!(recipe.nconfigs() > 0, "the seed IR offers tensor tile grids");
        // The 64x32x16 lane grid is among the configs: split then match.
        let mut found = false;
        for c in 0..recipe.nconfigs() {
            let mut probe = k.clone();
            recipe.apply(&mut probe, c);
            probe.verify();
            let wmmas = probe.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
            if wmmas == 32 {
                found = true;
            }
        }
        assert!(found, "the 4x4 lane grid tensorizes");
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

        // Both operands are staged: one shared f16 tile each, and the wmma's
        // fragment slots read them instead of global memory.
        let locals: Vec<OpId> = k
            .iter_unordered()
            .filter_map(|(id, op)| match op {
                Op::Storage { dtype: DType::F16, scope: MemScope::Local, .. } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(locals.len(), 2, "A and B each get a shared tile");

        let wmma = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Wmma { .. }).then_some(id))
            .expect("wmma present");
        let Op::Wmma { a: a_frag, b: b_frag, .. } = *k.at(wmma) else { unreachable!() };
        for operand in [a_frag, b_frag] {
            let Op::Stack { ref ops } = *k.at(operand) else { panic!("operand is not a stack") };
            for &slot in ops.iter() {
                let Op::Load { src, .. } = *k.at(slot) else { panic!("slot is not a load") };
                assert!(locals.contains(&src), "fragment slot still reads global memory");
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
