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
    DType, Map, Set,
    backend::{FragAxis, FragMap, MMAVariant},
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind, autotune::Optimization, split::SplitMode},
    shape::Dim,
};

/// One scalar matmul region and the instruction variant it will be built
/// with: the region's range and the affine views of all three operands.
#[derive(Debug, Clone)]
struct Region {
    variant: MMAVariant,
    init_store: OpId,
    /// The update store inside the K loop that writes `acc + prod`.
    update: OpId,
    /// Every output store that reads `acc` through a cast chain, with the
    /// store's destination and that chain (outermost cast first). Collected by
    /// the scan so `apply` never has to look at the IR again.
    epilogue: Vec<(OpId, OpId, Vec<crate::DType>)>,

    a_src: OpId,
    a_m: Dim,
    a_k: Dim,
    b_src: OpId,
    b_n: Dim,
    b_k: Dim,
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

/// Warp layouts a tensorized block may use, as `(m, n, k)` warps over the
/// output rows, the output columns, and the contraction. One axis at a time:
/// a warp grid owns disjoint output sub-tiles (nothing is summed), a K split
/// sums the per-warp partials in shared memory.
pub(crate) const WARP_LAYOUTS: [(u16, u16, u16); 13] = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (4, 1, 1),
    (2, 2, 1),
    (1, 4, 1),
    (8, 1, 1),
    (4, 2, 1),
    (2, 4, 1),
    (1, 8, 1),
    (1, 1, 2),
    (1, 1, 4),
    (1, 1, 8),
];

/// One tensorization candidate: a matched region, a pre-tiled shape, or a tile
/// grid this plan builds itself, at one warp layout, with the operands staged
/// through shared memory or read straight from global memory.
#[derive(Debug, Clone, Copy)]
struct Plan {
    /// A region the kernel already has, into [`Tensorize::regions`].
    region: Option<usize>,
    /// A shape whose lanes a split already built, into [`Tensorize::tiled`].
    tiled: Option<usize>,
    /// A tile grid this plan builds itself: the two output ranges, the K loop,
    /// the lane factors to split them by, and the instruction tile those
    /// factors are multiples of.
    grid: Option<(OpId, OpId, OpId, u16, u16, u16, Dim, Dim, Dim)>,
    layout: (u16, u16, u16),
    /// Stage the operands through shared memory (`ldmatrix`) instead of
    /// reading their fragments from global memory.
    staged: bool,
}

/// The tensorization optimization: one config per candidate plan.
#[derive(Debug)]
struct Tensorize {
    plans: Vec<Plan>,
    regions: Vec<Region>,
    tiled: Vec<Tiled>,
}

impl Optimization for Tensorize {
    fn nconfigs(&self) -> u64 {
        self.plans.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_tensorize apply");
        let Some(plan) = self.plans.get(config as usize) else {
            return;
        };
        if let Some((m, n, k, m_lanes, n_lanes, k_lanes, fm, fn_, fk)) = plan.grid {
            // The grid is this plan's own state: the predicates proved every
            // factor divides, so the splits apply. No clone, no probe.
            kernel.split(m, SplitMode::Unroll, m_lanes);
            kernel.split(n, SplitMode::Unroll, n_lanes);
            kernel.split(k, SplitMode::Unroll, k_lanes);
            // The tiled form is derived from the kernel as it now stands —
            // its range groups and its loop — and the grid says which tile
            // extents it must have.
            let want = (Dim::from(m_lanes) / fm, Dim::from(n_lanes) / fn_, Dim::from(k_lanes) / fk);
            let (_, forms) = kernel.tensor_forms();
            let mut emitted = 0usize;
            for t in &forms {
                if (t.mt, t.nt, t.kt) == want {
                    tile_to_wmma(kernel, t, plan.staged, plan.layout);
                    emitted += 1;
                }
            }
            assert!(emitted > 0, "the enumerated tile grid must be derivable from its own splits");
            kernel.verify();
            return;
        }
        let staged = plan.staged;
        let layout = plan.layout;
        if let Some(i) = plan.tiled {
            tile_to_wmma(kernel, &self.tiled[i], staged, layout);
            return;
        }
        let Some(region) = plan.region else {
            return;
        };
        let r = &self.regions[region];
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
        // Staged form needs both K views contiguous for a vector fill.
        let staged = staged && r.a_k == 1 && r.b_k == 1;
        // Staged form: one cooperative shared tile per operand, refilled at
        // every K step. The tile is k-block-major (`((k/8)*(rows/8) + (m/8))*64
        // + (m%8)*8 + (k%8)`), so it packs to `rows * k` halves with no padding
        // and needs none: an eight-half row of a block is 16 B and the rows of
        // a block walk the banks.
        let (a_tile, b_tile) = if staged {
            (insert_local_f16_storage(kernel, fm * fk), insert_local_f16_storage(kernel, fn_ * fk))
        } else {
            (OpId::NULL, OpId::NULL)
        };
        let threads = Dim::from(warp_size);

        // One mma per `fk` of K.
        let rounds = r.k / fk;
        kernel.loop_over(rounds, |k, kv| {
            let fk_c = k.const_idx(fk);
            let k_base = k.mul(kv, fk_c);
            let (a_frag, b_frag) = if staged {
                let a_k = k.mul(k_base, r.a_k);
                let a_base = k.mad(m_base, r.a_m, a_k);
                let b_k = k.mul(k_base, r.b_k);
                let b_base = k.mad(n_base, r.b_n, b_k);
                let (asl, av) = coop_fill_load(k, OpId::NULL, lane, r.a_src, a_base, r.a_m, r.a_k, fm, fk, threads);
                coop_fill_store(k, OpId::NULL, a_tile, OpId::NULL, asl, av);
                let (bsl, bv) = coop_fill_load(k, OpId::NULL, lane, r.b_src, b_base, r.b_n, r.b_k, fn_, fk, threads);
                coop_fill_store(k, OpId::NULL, b_tile, OpId::NULL, bsl, bv);
                k.push_back(Op::Barrier);
                let a = coop_ldmatrix(k, OpId::NULL, lane, a_tile, fm, fk, fm, 0, fk / 8, OpId::NULL, OpId::NULL);
                let b = coop_ldmatrix(k, OpId::NULL, lane, b_tile, fn_, fk, fn_, 0, fk / 8, OpId::NULL, OpId::NULL);
                (a, b)
            } else {
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
                (a_frag, k.stack(&b_elems))
            };
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
            if staged {
                // The next fill must not overwrite a tile the lanes are still
                // reading.
                k.push_back(Op::Barrier);
            }
        });

        // Scatter the accumulator: each lane writes its own fragment slots,
        // once per C output. The stores and cast chains come from the scan
        // that accepted the region; nothing is derived here.
        for (_, dst, chain) in &r.epilogue {
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

        // The scalar accumulation is obsolete: the scatter above writes the
        // wmma values. Delete the stores that rooted it — the init, the loop
        // update, and the epilogue stores just replaced — and let DCE take the
        // accumulator reads, slot arithmetic and the storage itself. Deleting
        // the whole init..final span by hand would also delete another
        // region's storage when two matmuls interleave in the IR.
        for store in r.epilogue.iter().map(|(store, _, _)| *store).chain([r.init_store, r.update]) {
            kernel.remove_op(store);
        }
        kernel.dead_code_elimination();
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
fn slot_addr(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    map: &FragMap,
    base: OpId,
    sr: Dim,
    sc: Dim,
    slot: (i16, i16),
    warp: u32,
) -> OpId {
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
fn acc_index(kernel: &mut Kernel, before: OpId, nt: Dim, slots: Dim, tm: Dim, tn: Dim, slot: usize) -> OpId {
    let idx = (tm * nt + tn) * slots + Dim::from(slot as i64);
    kernel.insert_before(before, Op::Const(Constant::idx(idx)))
}

/// A fragment operand's slot values, pairing slots that are contiguous in
/// memory into one vector load (the `half2`/`half4` the instruction consumes).
#[allow(clippy::too_many_arguments)]
fn frag_elems(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    map: &FragMap,
    src: OpId,
    base: OpId,
    sr: Dim,
    sc: Dim,
    warp: u32,
) -> Vec<OpId> {
    /// f16 halves per vector access: 8 bytes, the `ldg8` the reference's
    /// direct (unstage) kernel loads fragments with.
    const VEC: usize = 4;
    let slots = &map.slots;
    let mut out = Vec::with_capacity(slots.len());
    let mut i = 0;
    while i < slots.len() {
        // The longest run of slots that is contiguous in memory, up to one
        // vector access. Its innermost lane is the vector width:
        // `devectorize` keeps it and the backend renders one vector load,
        // which the `Stack` then reads lane by lane through `Index`.
        let mut n = 1;
        while i + n < slots.len() && n < VEC {
            let a = slots[i + n - 1];
            let b = slots[i + n];
            let delta = Dim::from(b.0 - a.0) * sr + Dim::from(b.1 - a.1) * sc;
            if delta != 1 {
                break;
            }
            n += 1;
        }
        let addr = slot_addr(kernel, before, lane, map, base, sr, sc, slots[i], warp);
        if n == 1 {
            out.push(kernel.insert_before(before, Op::Load { src, src_idx: addr }));
        } else {
            let vector = kernel.insert_before(before, Op::Unroll { axis: 0, len: n as u16 });
            let idx = kernel.insert_before(before, Op::Binary { x: addr, y: vector, bop: BOp::Add });
            let value = kernel.insert_before(before, Op::Load { src, src_idx: idx });
            for lane_idx in 0..n {
                out.push(kernel.insert_before(before, Op::Index { vec: value, idx: lane_idx }));
            }
        }
        i += n;
    }
    out
}

/// Insert a local storage after the kernel's parameters (global params must
/// precede local storages, the order `verify` expects).
fn insert_local_storage(kernel: &mut Kernel, dtype: DType, len: Dim) -> OpId {
    insert_scoped_storage(kernel, dtype, MemScope::Local, len)
}

/// Insert a storage of any scope after the kernel's parameters: global params
/// must precede every storage, the order `verify` and the dtype walk expect.
fn insert_scoped_storage(kernel: &mut Kernel, dtype: DType, scope: MemScope, len: Dim) -> OpId {
    let mut at = kernel.head;
    while matches!(kernel.at(at), Op::Param { .. }) {
        at = kernel.next_op(at);
    }
    kernel.insert_before(at, Op::Storage { dtype, scope, len, id: 0 })
}

/// Insert a local f16 storage after the kernel's parameters (global params
/// must precede local storages, the order `verify` expects).
fn insert_local_f16_storage(kernel: &mut Kernel, len: Dim) -> OpId {
    insert_local_storage(kernel, DType::F16, len)
}

/// Cooperatively fill a `rows x chunk` f16 tile with the global view `(src,
/// base, sr, sc)`: every lane copies one `half2` of a row, so a warp moves a
/// whole row (or several) per step with vectorized, coalesced accesses.
/// `stride` is the padded row length in halves. Requires the K view to be
/// contiguous (`sc == 1`).
#[allow(clippy::too_many_arguments)]
fn coop_fill_load(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    src: OpId,
    base: OpId,
    sr: Dim,
    sc: Dim,
    rows: Dim,
    chunk: Dim,
    threads: Dim,
) -> (OpId, OpId) {
    // The widest vector access this tile can use: 16 bytes (`half8`) where
    // the row and the thread count divide into it, then 8 bytes (`half4`),
    // then one `half2`. This is the width the reference's cooperative fill
    // issues (`LDG.E.128`).
    let vec: Dim = [8, 4, 2]
        .into_iter()
        .filter(|v| chunk % Dim::from(*v) == 0)
        .map(Dim::from)
        .find(|v| rows * (chunk / *v) % threads == 0)
        .unwrap_or(2);
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    debug_assert_eq!(sc, 1, "cooperative fill: the K view must be contiguous");
    debug_assert_eq!(chunk % vec, 0, "cooperative fill: K chunk not a whole number of vectors");
    let per_row = chunk / vec;
    debug_assert_eq!(rows * per_row % threads, 0, "cooperative fill: vector count not divisible by the threads");
    // One vector access per step, with the steps emitted rather than laned:
    // a step *lane* makes the value a two-dimensional `[steps x vec]`, which
    // no backend renders as one vector access, and `devectorize` then unfolds
    // One vector load and one vector store, with the step a thread walks down
    // the tile as a lane: `devectorize` keeps the innermost lane (the vector
    // width, which the backend renders as one access) and duplicates the step
    // lane, so the fill is written once here instead of once per step.
    let iters = rows * per_row / threads;
    // The vector width is the innermost lane (axis 1: the access's own
    // elements), the step is the outer one (axis 0: copies of the access).
    let vector = put(kernel, Op::Unroll { axis: 1, len: u16::try_from(vec).expect("fill vector width") });
    let prc = put(kernel, Op::Const(Constant::idx(per_row)));
    let vc = put(kernel, Op::Const(Constant::idx(vec)));
    let step = if iters > 1 {
        let lanes = put(kernel, Op::Unroll { axis: 0, len: u16::try_from(iters).expect("fill step count") });
        let per = put(kernel, Op::Const(Constant::idx(threads / per_row)));
        put(kernel, Op::Binary { x: lanes, y: per, bop: BOp::Mul })
    } else {
        put(kernel, Op::Const(Constant::idx(0i64)))
    };
    // The thread's own row and column from its linear id: a shift and a mask
    // when the vectors per row is a power of two — the reference's
    // `LOP3 tid&7` / `SHF <<3` — and never a signed division, which lowers to a
    // call sequence. The id is never negative, so the bitwise forms are exact.
    let pr_n = u32::try_from(per_row).expect("cooperative fill: vectors per row");
    let (tid_row, tid_col) = if pr_n.is_power_of_two() {
        let sh = put(kernel, Op::Const(Constant::idx(i64::from(pr_n.trailing_zeros()))));
        let row = put(kernel, Op::Binary { x: lane, y: sh, bop: BOp::BitShiftRight });
        let m = put(kernel, Op::Const(Constant::idx(i64::from(pr_n) - 1)));
        let col = put(kernel, Op::Binary { x: lane, y: m, bop: BOp::BitAnd });
        (row, col)
    } else {
        let row = put(kernel, Op::Binary { x: lane, y: prc, bop: BOp::Div });
        let col = put(kernel, Op::Binary { x: lane, y: prc, bop: BOp::Mod });
        (row, col)
    };
    // The row is the thread's own row plus the step it walks: two terms, so the
    // step's share stays a constant and folds into the access' immediate —
    // `(step + tid_row) * stride` puts the multiply on a sum, and every step
    // then needs a register of its own instead of `[Rbase+imm]`. (`step` is
    // already the row offset: the lane times the rows one step covers.)
    let rrow = step;
    let rterm_v = if sr == 1 {
        tid_row
    } else {
        let c = put(kernel, Op::Const(Constant::idx(sr)));
        put(kernel, Op::Binary { x: tid_row, y: c, bop: BOp::Mul })
    };
    let cterm = put(kernel, Op::Binary { x: tid_col, y: vc, bop: BOp::Mul });
    let rterm_s = if sr == 1 {
        rrow
    } else {
        let c = put(kernel, Op::Const(Constant::idx(sr)));
        put(kernel, Op::Binary { x: rrow, y: c, bop: BOp::Mul })
    };
    let gidx = put(kernel, Op::Binary { x: base, y: rterm_v, bop: BOp::Add });
    let gidx = put(kernel, Op::Binary { x: gidx, y: cterm, bop: BOp::Add });
    let gidx = put(kernel, Op::Binary { x: gidx, y: rterm_s, bop: BOp::Add });
    let gidx = put(kernel, Op::Binary { x: gidx, y: vector, bop: BOp::Add });
    let value = put(kernel, Op::Load { src, src_idx: gidx });
    // The shared slot is the k-block-major tile layout —
    // `((k/8)*(rows/8) + (m/8))*64 + (m%8)*8 + (k%8)` — so a vector is one
    // eight-half row of a k-block and its lanes are contiguous. Every other
    // term is a shift or a mask, and the tile packs to `rows * chunk` halves
    // with no padding: the reference's 16 KB per operand, and the 32 KB that
    // keeps two blocks resident per SM.
    let row = put(kernel, Op::Binary { x: tid_row, y: rrow, bop: BOp::Add });
    let three = put(kernel, Op::Const(Constant::idx(3i64)));
    let seven = put(kernel, Op::Const(Constant::idx(7i64)));
    let mblk = put(kernel, Op::Binary { x: row, y: three, bop: BOp::BitShiftRight });
    let kblk = put(kernel, Op::Binary { x: cterm, y: three, bop: BOp::BitShiftRight });
    // Row swizzle: the eight lanes of a store share a row of their 128-byte
    // block and differ only in their k-block, so with a linear index every one
    // of them lands on banks 0-3 (measured: 8-way conflict on each of the 16
    // fill stores, `mio_throttle` the top stall). XORing the row field with the
    // unit's k-block spreads them over the banks, and `coop_ldmatrix` mirrors it
    // with the k-block of the unit it reads.
    let mrow = put(kernel, Op::Binary { x: row, y: seven, bop: BOp::BitAnd });
    let swz = put(kernel, Op::Binary { x: kblk, y: seven, bop: BOp::BitAnd });
    let mrow = put(kernel, Op::Binary { x: mrow, y: swz, bop: BOp::BitXor });
    let mrow = put(kernel, Op::Binary { x: mrow, y: three, bop: BOp::BitShiftLeft });
    // The column's offset inside its eight-half block: the slot is
    // `... + (k%8)`, and `vector` only covers the vector's own lanes. Without
    // this term every thread of a row writes the same eight halves, so a tile
    // holds one vector per row instead of the whole window.
    let kin = put(kernel, Op::Binary { x: cterm, y: seven, bop: BOp::BitAnd });
    let rblk = put(kernel, Op::Const(Constant::idx(rows / 8)));
    let kblk = put(kernel, Op::Binary { x: kblk, y: rblk, bop: BOp::Mul });
    let blk = put(kernel, Op::Binary { x: kblk, y: mblk, bop: BOp::Add });
    let pack = put(kernel, Op::Const(Constant::idx(64i64)));
    let slot = put(kernel, Op::Binary { x: blk, y: pack, bop: BOp::Mul });
    let slot = put(kernel, Op::Binary { x: slot, y: mrow, bop: BOp::Add });
    let slot = put(kernel, Op::Binary { x: slot, y: kin, bop: BOp::Add });
    let slot = put(kernel, Op::Binary { x: slot, y: vector, bop: BOp::Add });
    (slot, value)
}

/// The store half of a staged fill: the slot [`coop_fill_load`] produced, moved
/// to the stage `slot_base` and written. Split from the load so the loads can
/// be issued before the window's compute and the stores after it — that gap is
/// what hides the global latency, and it is the shape the reference's body has
/// (`LDG` early, `STS` late, before the barrier).
fn coop_fill_store(kernel: &mut Kernel, before: OpId, tile: OpId, slot_base: OpId, slot: OpId, value: OpId) {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    let slot = if slot_base.is_null() {
        slot
    } else {
        put(kernel, Op::Binary { x: slot, y: slot_base, bop: BOp::Add })
    };
    put(kernel, Op::Store { dst: tile, dst_idx: slot, src: value });
}

/// The `ldmatrix` that reads the `k_offset`-th instruction sub-tile of a
/// staged `rows x chunk` tile in the fragment layout of the MMA: `frags` 8x8
/// b16 tiles, one per group of eight lanes, each lane naming the row it
/// reads. A `m16n8k8` A tile is `x2` (two 8-row matrices); with two K steps
/// per load (`k = 2 * 8`) it is `x4` and B's is `x2`.
/// The lane-dependent half of an `ldmatrix` address: the row this lane reads
/// in its eight-lane group, and — with more than one K step in the load — the
/// next eight columns for the lane groups above `rows`. Constant across the
/// sub-tile offsets, so one per tile (the reference keeps it in one register
/// and reaches the other sub-tiles with immediates).
fn ldmatrix_lane_base(kernel: &mut Kernel, before: OpId, lane: OpId, rows: Dim, k: Dim, tile_rows: Dim) -> OpId {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    // The lane's row inside the loaded region, and — with more than one K step
    // in the load — which 8-K block the upper lane groups read.
    let rconst = put(kernel, Op::Const(Constant::idx(rows)));
    let row = put(kernel, Op::Binary { x: lane, y: rconst, bop: BOp::Mod });
    // k-block-major: the row's block is `(m/8)`, its eight-half row `m%8`.
    let three = put(kernel, Op::Const(Constant::idx(3i64)));
    let seven = put(kernel, Op::Const(Constant::idx(7i64)));
    let mblk = put(kernel, Op::Binary { x: row, y: three, bop: BOp::BitShiftRight });
    let mrow = put(kernel, Op::Binary { x: row, y: seven, bop: BOp::BitAnd });
    let mrow = put(kernel, Op::Binary { x: mrow, y: three, bop: BOp::BitShiftLeft });
    let pack = put(kernel, Op::Const(Constant::idx(64i64)));
    let mut addr = put(kernel, Op::Binary { x: mblk, y: pack, bop: BOp::Mul });
    addr = put(kernel, Op::Binary { x: addr, y: mrow, bop: BOp::Add });
    if k / 8 > 1 {
        let seg = put(kernel, Op::Binary { x: lane, y: rconst, bop: BOp::Div });
        let kstride = put(kernel, Op::Const(Constant::idx(tile_rows / 8 * 64)));
        let ko = put(kernel, Op::Binary { x: seg, y: kstride, bop: BOp::Mul });
        addr = put(kernel, Op::Binary { x: addr, y: ko, bop: BOp::Add });
    }
    addr
}

#[allow(clippy::too_many_arguments)]
fn coop_ldmatrix(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    tile: OpId,
    rows: Dim,
    k: Dim,
    tile_rows: Dim,
    k_offset: Dim,
    kblocks: Dim,
    slot_base: OpId,
    lane_base: OpId,
) -> OpId {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    let frags = u8::try_from(rows / 8 * (k / 8)).expect("ldmatrix: too many fragments");
    // The lane's own row/column offset is the same for every sub-tile of the
    // tile: reuse the caller's (one per tile) instead of rebuilding it per
    // load. `OpId::NULL` computes it here for a single-use load.
    let addr = if lane_base.is_null() {
        ldmatrix_lane_base(kernel, before, lane, rows, k, tile_rows)
    } else {
        lane_base
    };
    // Order matters for the *SASS*: the two varying-but-loop-invariant terms
    // are summed first and the K offset, a constant per load, is added last, so
    // the address is `base + immediate` and ptxas folds it into the
    // instruction (`LDSM [R200+UR8+0x1000]`, as the reference has it). Adding
    // the constant in the middle gives every load its own register instead.
    let addr = if slot_base.is_null() {
        addr
    } else {
        put(kernel, Op::Binary { x: addr, y: slot_base, bop: BOp::Add })
    };
    // A K offset in halves is `k_offset / 8` k-blocks, each one 64-half block
    // per eight tile rows: the k-block-major pack strides them by
    // `tile_rows / 8 * 64`. It is folded in before the swizzle: the row an
    // element lives in is chosen by the k-block of *this* unit, so the key has
    // to see the sub-tile's offset, not just the tile's base.
    let off = put(kernel, Op::Const(Constant::idx(k_offset * (tile_rows / 8 * 8))));
    let addr = put(kernel, Op::Binary { x: addr, y: off, bop: BOp::Add });
    // Row swizzle, the read half. The fill XORs the row field with the k-block
    // it wrote, so the read must address the row the element actually lives in:
    // XOR the same field (8 halves per row, bits 3..5 of the element index) with
    // the k-block of the unit being read. Reading it back off the index itself
    // (`index / halves per k-block`) is what keeps this identical to the fill's
    // key without either side restating the other's lane geometry — and since
    // every shared address here is loop-invariant, it is computed once, not per
    // iteration.
    let one = put(kernel, Op::Const(Constant::idx(1i64)));
    let eight = put(kernel, Op::Const(Constant::idx(8i64)));
    let kbh = put(kernel, Op::Const(Constant::idx(tile_rows / 8 * 64)));
    let key = put(kernel, Op::Binary { x: addr, y: kbh, bop: BOp::Div });
    // The key must be the k-block *inside the operand's tile*, the way the fill
    // writes it: the absolute address also carries the tile's base, so masking
    // by the tile's own k-block count is what keeps the read key equal to the
    // store key (`kt = 8` only agreed by coincidence, because a 16 KB tile is a
    // whole multiple of eight k-blocks and a 8 KB one is not).
    let nkb = put(kernel, Op::Const(Constant::idx(kblocks.max(1))));
    let kmask = put(kernel, Op::Binary { x: nkb, y: one, bop: BOp::Sub });
    let key = put(kernel, Op::Binary { x: key, y: kmask, bop: BOp::BitAnd });
    let swz = put(kernel, Op::Binary { x: key, y: eight, bop: BOp::Mul });
    let addr = put(kernel, Op::Binary { x: addr, y: swz, bop: BOp::BitXor });
    put(kernel, Op::PTX(Box::new(PTXOp::VLdMatrix { frags, src: tile, addr, trans: false })))
}

fn acc_frag_index(
    kernel: &mut Kernel,
    t: &Tiled,
    v: &MMAVariant,
    nt: Dim,
    tm: OpId,
    tn: OpId,
    slots_axis: u32,
    at: OpId,
) -> OpId {
    // The C fragment's slots are consecutive in the accumulator, so the whole
    // fragment is one vector access: the lane `Unroll(n)` is the index vector
    // `{0, 1, .., n-1}` on top of the fragment's base slot. `tm`/`tn` are ops,
    // so a caller may pass a tile lane instead of a constant.
    let slots = u16::try_from(v.c.slots.len()).expect("C fragment slot count");
    let ntc = kernel.insert_const_idx_before(at, nt);
    let base = kernel.insert_before(at, Op::Binary { x: tm, y: ntc, bop: BOp::Mul });
    let base = kernel.insert_before(at, Op::Binary { x: base, y: tn, bop: BOp::Add });
    let sc = kernel.insert_const_idx_before(at, t.slots);
    let base = kernel.insert_before(at, Op::Binary { x: base, y: sc, bop: BOp::Mul });
    let lanes = kernel.insert_before(at, Op::Unroll { axis: slots_axis, len: slots });
    kernel.insert_before(at, Op::Binary { x: base, y: lanes, bop: BOp::Add })
}

/// Emit a matched tiled region as wmma instructions, in place: the
/// accumulator storage keeps its identity, its accesses are rewritten in
/// fragment order, and the split's reduce/update and the output stores are
/// replaced (DCE drops their cones).
fn tile_to_wmma(kernel: &mut Kernel, t: &Tiled, staged: bool, layout: (u16, u16, u16)) {
    let v = &t.variant;
    let (fm, fn_, fk) = {
        let (m, n, k) = v.dims.decompose_mnk();
        (m as Dim, n as Dim, k as Dim)
    };
    let warp = u32::from(kernel.device_info().warp_size);
    let Op::Loop { len: loop_len_id } = *kernel.at(t.loop_id) else {
        unreachable!("tiled tensorize: the K loop is not a loop")
    };
    let chunk = t.kt * fk;
    // The block-wide form stages a window of two K steps (one `ldmatrix` per
    // row group covering both) in two buffers, so the output tile is not
    // limited by the whole K chunk living in shared memory.
    // The staged window: the two K steps one `ldmatrix` covers, in two
    // buffers. A wider window costs shared memory and measured slower.
    let window_width = if t.kt % 2 == 0 { fk * 2 } else { fk };
    // The block's warp layout: `m` warps over the output rows and `n` over the
    // output columns (a warp grid — every warp owns its sub-tile, so nothing
    // is summed across warps), or `k` warps over the contraction (the partial
    // accumulators are summed through shared memory). One axis at a time.
    let (wm, wn, wk) = layout;
    let cols = t.nt * fn_;
    let rows = if fn_ > fm { fn_ } else { fm };
    let staged_form = staged && t.a_k == 1 && t.b_k == 1;
    // The staged form holds one warp-private tile plus, for a K split, the
    // combine buffer `[output column][warp * rows + row]`, which needs the
    // block's output rows (`t.nt * fn_`) to be whole warp-wide stores.
    let combine = wk > 1;
    if combine && cols % Dim::from(warp) != 0 {
        // The combine epilogue stores one warp of rows per column.
        return;
    }
    let staged_bytes = |nw: Dim| {
        // The block-wide form holds two stages of a two-K-step window; a
        // split-K warp holds its own tile and merges through the combine
        // buffer.
        let tile = if combine {
            nw * rows * chunk * 2
        } else {
            (t.mt * fm + t.nt * fn_) * chunk * 2
        };
        if combine {
            tile + t.mt * fm * (nw * cols + 4) * 4
        } else {
            tile
        }
    };
    // A layout whose K slice does not fit the loop, or whose warp grid does
    // not split the tile grid evenly, falls back to the single-warp form. The
    // loop length is a symbolic expression: its divisibility is decided on the
    // expression, and only the K-split warp count needs a value at all.
    let (wm, wn, wk) = if (wk > 1 && !kernel.is_sym_multiple(loop_len_id, Dim::from(wk)))
        || (wm > 1 && t.mt % Dim::from(wm) != 0)
        || (wn > 1 && t.nt % Dim::from(wn) != 0)
    {
        (1, 1, 1)
    } else {
        (wm, wn, wk)
    };
    // The tiles one warp owns: a warp grid hands each warp a contiguous band
    // of the block's tile grid.
    let mt_w = t.mt / Dim::from(if wm > 1 { wm } else { 1 });
    let nt_w = t.nt / Dim::from(if wn > 1 { wn } else { 1 });
    let warps = u32::from(wm) * u32::from(wn) * u32::from(wk);
    let combine = wk > 1;
    let mut warp_var = OpId::NULL;
    let mut warp_off_a = OpId::NULL;
    let mut warp_off_b = OpId::NULL;
    let mut warp_off_m = OpId::NULL;
    let mut warp_off_n = OpId::NULL;
    // The staged form slices K round-robin: warp `y` takes chunks `y`,
    // `y + n`, ... of one tile step each, so the warps' loads of a row stay
    // in one window of it. A contiguous slice otherwise.
    let interleave = staged_form && combine;
    if warps > 1 {
        warp_var = kernel.insert_before(t.loop_id, Op::Range { axis: 1, kind: RangeKind::Local(warps) });
    }
    if combine {
        // A known loop length keeps its folded quotient; a symbolic one
        // carries the exact `Div` (the gate above proved the division exact).
        let sub = match kernel.resolve_const(loop_len_id).and_then(Constant::as_dim) {
            Some(l) => kernel.insert_const_idx_before(t.loop_id, l / Dim::from(wk)),
            None => {
                let wkc = kernel.insert_const_idx_before(t.loop_id, Dim::from(wk));
                kernel.insert_before(t.loop_id, Op::Binary { x: loop_len_id, y: wkc, bop: BOp::Div })
            }
        };
        kernel.ops[t.loop_id].op = Op::Loop { len: sub };
        let a_step = t.a_fixed.iter().find(|(v, _)| *v == t.loop_id).map(|&(_, s)| s).unwrap_or(0);
        let b_step = t.b_fixed.iter().find(|(v, _)| *v == t.loop_id).map(|&(_, s)| s).unwrap_or(0);
        let na = if interleave {
            kernel.insert_const_idx_before(t.loop_id, Dim::from(1))
        } else {
            sub
        };
        let ca = {
            let s = kernel.insert_const_idx_before(t.loop_id, a_step);
            kernel.insert_before(t.loop_id, Op::Binary { x: na, y: s, bop: BOp::Mul })
        };
        warp_off_a = kernel.insert_before(t.loop_id, Op::Binary { x: warp_var, y: ca, bop: BOp::Mul });
        let cb = {
            let s = kernel.insert_const_idx_before(t.loop_id, b_step);
            kernel.insert_before(t.loop_id, Op::Binary { x: na, y: s, bop: BOp::Mul })
        };
        warp_off_b = kernel.insert_before(t.loop_id, Op::Binary { x: warp_var, y: cb, bop: BOp::Mul });
    }
    // A warp grid moves each warp's output window: `wi = warp % wm` selects
    // the row band, `wj = warp / wm` the column band. The offsets are element
    // counts (rows, columns); each use scales them by its own stride.
    if wm > 1 {
        let c = kernel.insert_const_idx_before(t.loop_id, Dim::from(wm));
        let wi = kernel.insert_before(t.loop_id, Op::Binary { x: warp_var, y: c, bop: BOp::Mod });
        let s = kernel.insert_const_idx_before(t.loop_id, mt_w * fm);
        warp_off_m = kernel.insert_before(t.loop_id, Op::Binary { x: wi, y: s, bop: BOp::Mul });
    }
    if wn > 1 {
        let wj = if wm > 1 {
            let c = kernel.insert_const_idx_before(t.loop_id, Dim::from(wm));
            kernel.insert_before(t.loop_id, Op::Binary { x: warp_var, y: c, bop: BOp::Div })
        } else {
            warp_var
        };
        let s = kernel.insert_const_idx_before(t.loop_id, nt_w * fn_);
        warp_off_n = kernel.insert_before(t.loop_id, Op::Binary { x: wj, y: s, bop: BOp::Mul });
    }
    // The same offsets as flat addresses: rows scale by A's row stride, the
    // columns by B's. The store side scales them by the output's strides.
    let scale_by = |kernel: &mut Kernel, off: OpId, stride: Dim| -> OpId {
        if off.is_null() {
            return OpId::NULL;
        }
        let s = kernel.insert_const_idx_before(t.loop_id, stride);
        kernel.insert_before(t.loop_id, Op::Binary { x: off, y: s, bop: BOp::Mul })
    };
    let a_band = scale_by(kernel, warp_off_m, t.a_m);
    let b_band = scale_by(kernel, warp_off_n, t.b_n);
    // Interleaving walks the loop `wk` chunks at a time, so the loop
    // variable's own step grows by `wk`.
    let scale_loop = |fixed: &[(OpId, Dim)]| -> Vec<(OpId, Dim)> {
        let n = Dim::from(wk);
        fixed.iter().map(|&(v, s)| (v, if v == t.loop_id && interleave { s * n } else { s })).collect()
    };
    let a_fixed = scale_loop(&t.a_fixed);
    let b_fixed = scale_loop(&t.b_fixed);
    // Add a base's warp offsets (K slice and/or output sub-tile) in place.
    let shift = |kernel: &mut Kernel, before: OpId, base: OpId, offs: [OpId; 2]| -> OpId {
        let mut addr = base;
        for off in offs {
            if !off.is_null() {
                addr = kernel.insert_before(before, Op::Binary { x: addr, y: off, bop: BOp::Add });
            }
        }
        addr
    };
    let lane = kernel.insert_before(t.loop_id, Op::Range { axis: 0, kind: RangeKind::Local(warp) });
    let nw = Dim::from(warps);
    // Shared memory the tile and the combine buffer need, in bytes; fall back
    // to the direct form otherwise. A block-wide fill also needs the tile to
    // hold at least one vector per thread and to divide evenly among them.
    let threads = Dim::from(warp);
    let vecs = |rows: Dim| rows * (window_width / Dim::from(2));
    let all = nw * threads;
    let fill_fits = combine
        || (vecs(t.mt * fm) >= all && vecs(t.mt * fm) % all == 0) && (vecs(t.nt * fn_) >= all && vecs(t.nt * fn_) % all == 0);
    // The staged form's `ldmatrix` group is one lane over both operands and
    // every K pair, which needs the operands' loads to share shape and stride:
    // equal tile rows and half as many m tiles per warp as n tiles. Otherwise
    // the config runs the per-tile form.
    // A split-K warp owns a private tile and never forms the block-wide group.
    let lane_group = combine || (t.mt * fm == t.nt * fn_ && mt_w * Dim::from(2) == nt_w);
    let staged = staged_form && lane_group && fill_fits && staged_bytes(nw) <= kernel.device_info().local_mem_size;

    // The accumulator holds one register per (tile, slot): `m_len * n_len`
    // scalar elements would be 32x what the instruction layout needs.
    // A warp grid hands each warp its own band of the tile grid, so the
    // accumulator only holds that band: the whole grid's `mt * nt * slots`
    // would be eight times the registers a warp uses.
    let len = mt_w * nt_w * t.slots;
    // A thread's accumulators must fit the register file, and the unstage
    // form emits one body per tile grid entry: a grid too wide for the
    // registers or the IR is not a kernel.
    if !staged && (len > Dim::from(128) || t.mt * t.nt * t.kt > Dim::from(256)) {
        return;
    }
    if let Op::Storage { len: l, .. } = &mut kernel.ops[t.acc].op {
        *l = len;
    }
    // Zero it: the scalar init store wrote the whole old tile.
    let zero = match kernel.at(t.init) {
        Op::Store { src, .. } => *src,
        _ => unreachable!("tiled tensorize: init is not a store"),
    };
    // One store, zeroed through lanes: the fragment lane is duplicated (one
    // store per fragment of the accumulator) and the slot lane stays the
    // vector width, so the zeroing is written once for the whole storage.
    {
        let slots: Dim = Dim::from(t.variant.c.slots.len() as i64);
        let frags = len / slots;
        let fr =
            kernel.insert_before(t.init, Op::Unroll { axis: 0, len: u16::try_from(frags).expect("accumulator fragment count") });
        let sl = kernel.insert_before(t.init, Op::Unroll { axis: 1, len: u16::try_from(slots).expect("fragment slot count") });
        let sc = kernel.insert_const_idx_before(t.init, slots);
        let base = kernel.insert_before(t.init, Op::Binary { x: fr, y: sc, bop: BOp::Mul });
        let idx = kernel.insert_before(t.init, Op::Binary { x: base, y: sl, bop: BOp::Add });
        kernel.insert_before(t.init, Op::Store { dst: t.acc, dst_idx: idx, src: zero });
    }
    kernel.remove_op(t.init);

    if staged {
        // The reference's iteration: one block-wide shared tile per operand,
        // filled cooperatively by every thread from vector global loads, a
        // block barrier, then `ldmatrix` from each warp's own band of it and
        // the instructions. Every element is loaded from global once per
        // block and reused by every warp that needs it.
        let mt = mt_w as usize;
        let nt = nt_w as usize;
        let kt = t.kt as usize;
        let a_rows = if combine { fm } else { t.mt * fm };
        let b_rows = if combine { fn_ } else { t.nt * fn_ };
        // A split-K warp keeps its own tile, so the buffer holds one per warp.
        let copies = if combine { nw } else { Dim::from(1) };
        let a_tile = insert_local_f16_storage(kernel, copies * a_rows * chunk);
        let b_tile = insert_local_f16_storage(kernel, copies * b_rows * chunk);
        // Split-K: `warp_var` selects the warp's own tile.
        let per_warp = fm * chunk;
        let b_per_warp = fn_ * chunk;
        let own = |kernel: &mut Kernel, per: Dim| -> OpId {
            if warp_var.is_null() {
                kernel.insert_const_idx_before(t.update, 0i64)
            } else {
                let c = kernel.insert_const_idx_before(t.update, per);
                kernel.insert_before(t.update, Op::Binary { x: warp_var, y: c, bop: BOp::Mul })
            }
        };
        let (a_slot, b_slot) = (own(kernel, per_warp), own(kernel, b_per_warp));
        let sync = |kernel: &mut Kernel| {
            if combine {
                kernel.insert_before(t.update, Op::PTX(Box::new(PTXOp::WarpSync)));
            } else {
                kernel.insert_before(t.update, Op::Barrier);
            }
        };
        // The threads one fill covers: every thread of the block for the
        // block-wide tile, one warp otherwise.
        let (fill_lane, fill_threads) = if combine {
            (lane, threads)
        } else if warp_var.is_null() {
            (lane, threads)
        } else {
            let w = kernel.insert_const_idx_before(t.loop_id, Dim::from(warp));
            let base = kernel.insert_before(t.loop_id, Op::Binary { x: warp_var, y: w, bop: BOp::Mul });
            let tid = kernel.insert_before(t.loop_id, Op::Binary { x: base, y: lane, bop: BOp::Add });
            (tid, Dim::from(warps) * Dim::from(warp))
        };
        let a_slots = v.a.slots.len();
        let b_slots = v.b.slots.len();
        let mut a_frags = vec![vec![OpId::NULL; kt]; mt];
        // The `slots` halves of a `ldmatrix` result that start at `base`: one
        // instruction operand out of a load that covers several.
        let slice_at = |kernel: &mut Kernel, before: OpId, value: OpId, base: usize, slots: usize| -> OpId {
            let mut elems = Vec::with_capacity(slots);
            for i in 0..slots {
                elems.push(kernel.insert_before(before, Op::Index { vec: value, idx: base + i }));
            }
            kernel.insert_before(before, Op::Stack { ops: elems.into() })
        };
        let slice = |kernel: &mut Kernel, before: OpId, value: OpId, second: bool, slots: usize| -> OpId {
            slice_at(kernel, before, value, if second { slots } else { 0 }, slots)
        };
        if combine {
            sync(kernel);
            // A warp-private tile holds one operand sub-tile, refilled per
            // sub-tile and read back in full before the refill.
            for tm in 0..mt {
                let a_base = base_expr(kernel, t.update, &a_fixed, t.a_offset, tm as Dim * fm * t.a_m);
                let a_base = shift(kernel, t.update, a_base, [warp_off_a, a_band]);
                let sb = a_slot;
                sync(kernel);
                let (asl, av) =
                    coop_fill_load(kernel, t.update, fill_lane, t.a_src, a_base, t.a_m, t.a_k, fm, chunk, fill_threads);
                coop_fill_store(kernel, t.update, a_tile, sb, asl, av);
                sync(kernel);
                let mut tk = 0;
                while tk < kt {
                    if tk + 1 < kt {
                        let value = coop_ldmatrix(
                            kernel,
                            t.update,
                            lane,
                            a_tile,
                            fm,
                            fk * 2,
                            fm,
                            tk as Dim * fk,
                            chunk / 8,
                            sb,
                            OpId::NULL,
                        );
                        a_frags[tm][tk] = slice(kernel, t.update, value, false, a_slots);
                        a_frags[tm][tk + 1] = slice(kernel, t.update, value, true, a_slots);
                        tk += 2;
                    } else {
                        a_frags[tm][tk] =
                            coop_ldmatrix(kernel, t.update, lane, a_tile, fm, fk, fm, tk as Dim * fk, chunk / 8, sb, OpId::NULL);
                        tk += 1;
                    }
                }
            }
            for tn in 0..nt {
                let b_base = base_expr(kernel, t.update, &b_fixed, t.b_offset, tn as Dim * fn_ * t.b_n);
                let b_base = shift(kernel, t.update, b_base, [warp_off_b, b_band]);
                let sb = b_slot;
                sync(kernel);
                let (bsl, bv) =
                    coop_fill_load(kernel, t.update, fill_lane, t.b_src, b_base, t.b_n, t.b_k, fn_, chunk, fill_threads);
                coop_fill_store(kernel, t.update, b_tile, sb, bsl, bv);
                sync(kernel);
                // The accumulator stays in registers across the warp's K steps:
                // one read before them, one write after them, and each step's
                // instruction takes the previous step's result as its `c`. That
                // `c` is an in-place accumulate (the instruction overwrites it),
                // so the storage round-trip between steps disappears.
                let tmos: Vec<OpId> = (0..mt).map(|tm| kernel.insert_const_idx_before(t.update, tm as Dim)).collect();
                let tno = kernel.insert_const_idx_before(t.update, tn as Dim);
                let mut acc_idx = Vec::with_capacity(mt);
                let mut acc = Vec::with_capacity(mt);
                for &tmo in &tmos {
                    let idx = acc_frag_index(kernel, t, v, nt_w, tmo, tno, 0, t.update);
                    acc.push(kernel.insert_before(t.update, Op::Load { src: t.acc, src_idx: idx }));
                    acc_idx.push(idx);
                }
                let mut tk = 0;
                while tk < kt {
                    if tk + 1 < kt {
                        let value = coop_ldmatrix(
                            kernel,
                            t.update,
                            lane,
                            b_tile,
                            fn_,
                            fk * 2,
                            fn_,
                            tk as Dim * fk,
                            chunk / 8,
                            sb,
                            OpId::NULL,
                        );
                        for (step, frag) in [
                            (tk, slice(kernel, t.update, value, false, b_slots)),
                            (tk + 1, slice(kernel, t.update, value, true, b_slots)),
                        ] {
                            for tm in 0..mt {
                                acc[tm] = kernel.insert_before(
                                    t.update,
                                    Op::Wmma {
                                        dims: v.dims,
                                        layout: v.layout,
                                        dtype: v.dtype,
                                        a: a_frags[tm][step],
                                        b: frag,
                                        c: acc[tm],
                                    },
                                );
                            }
                        }
                        tk += 2;
                    } else {
                        let frag = coop_ldmatrix(
                            kernel,
                            t.update,
                            lane,
                            b_tile,
                            fn_,
                            fk,
                            fn_,
                            tk as Dim * fk,
                            chunk / 8,
                            sb,
                            OpId::NULL,
                        );
                        for tm in 0..mt {
                            acc[tm] = kernel.insert_before(
                                t.update,
                                Op::Wmma {
                                    dims: v.dims,
                                    layout: v.layout,
                                    dtype: v.dtype,
                                    a: a_frags[tm][tk],
                                    b: frag,
                                    c: acc[tm],
                                },
                            );
                        }
                        tk += 1;
                    }
                }
                for tm in 0..mt {
                    kernel.insert_before(t.update, Op::Store { dst: t.acc, dst_idx: acc_idx[tm], src: acc[tm] });
                }
            }
        } else {
            // One tile per operand, refilled in place: the reference's
            // mainloop. The whole K chunk is the window, so there is no window
            // loop and no second stage — the tile the `ldmatrix`s read is the
            // one the next chunk is written over, which is why the loads go out
            // at the top of the body, the stores land after the instructions,
            // and a barrier sits either side of them (read-protect, publish).
            let paired = kt >= 2;
            let win_steps = kt;
            let win_k = Dim::from(win_steps as i64) * fk;
            // k-block-major, so a tile is exactly `rows * k` halves: 16 KB per
            // operand, 32 KB for the pair, and no row padding at all.
            let a_rows = t.mt * fm;
            let b_rows = t.nt * fn_;
            let a_win = a_rows * win_k;
            let b_win = b_rows * win_k;
            // The K element a chunk covers, wrapped into the contraction: the
            // fill of the last chunk of an iteration is the first chunk of the
            // next one, so the loads are already in flight while the
            // instructions of the current chunk run.
            let a_step = a_fixed.iter().find(|(v, _)| *v == t.loop_id).map(|&(_, s)| s).unwrap_or(0);
            let b_step = b_fixed.iter().find(|(v, _)| *v == t.loop_id).map(|&(_, s)| s).unwrap_or(0);
            let a_rest: Vec<(OpId, Dim)> = a_fixed.iter().copied().filter(|(v, _)| *v != t.loop_id).collect();
            let b_rest: Vec<(OpId, Dim)> = b_fixed.iter().copied().filter(|(v, _)| *v != t.loop_id).collect();
            let a_origin = base_expr(kernel, t.update, &a_rest, t.a_offset, 0);
            let b_origin = base_expr(kernel, t.update, &b_rest, t.b_offset, 0);
            // One tile for both operands, A first: the reference's single 32 KB
            // buffer, with B one A-section after it, so the K pairs of both
            // operands are one lane of eight equal-strided `ldmatrix`es.
            let ab_buf = insert_local_f16_storage(kernel, a_win + b_win);
            let a_win_c = kernel.insert_const_idx_before(t.loop_id, a_win);
            // Prologue: the first chunk of the loop's first iteration.
            {
                let a0 = base_expr(kernel, t.loop_id, &a_rest, t.a_offset, 0);
                let b0 = base_expr(kernel, t.loop_id, &b_rest, t.b_offset, 0);
                let (asl, av) =
                    coop_fill_load(kernel, t.loop_id, fill_lane, t.a_src, a0, t.a_m, t.a_k, a_rows, win_k, fill_threads);
                coop_fill_store(kernel, t.loop_id, ab_buf, OpId::NULL, asl, av);
                let (bsl, bv) =
                    coop_fill_load(kernel, t.loop_id, fill_lane, t.b_src, b0, t.b_n, t.b_k, b_rows, win_k, fill_threads);
                coop_fill_store(kernel, t.loop_id, ab_buf, a_win_c, bsl, bv);
                // Every warp reads rows other threads wrote, so the prologue's
                // chunk has to be published before the first `ldmatrix`.
                kernel.insert_before(t.loop_id, Op::Barrier);
            }
            // The fill element is loop-carried, not rebuilt per chunk: the K
            // element the fill reads, in the operand's own stride, advanced by
            // one chunk per iteration and wrapped at the end of the contraction
            // (the last chunk of an iteration is the first of the next). The
            // reference advances one pointer per operand
            // (`IADD3 R130, P2, R130, c[0x0][0x178]`) instead of recomputing
            // `k * stride` from the loop counter.
            let a_off_st = insert_scoped_storage(kernel, DType::I64, MemScope::Register, Dim::from(1));
            let b_off_st = insert_scoped_storage(kernel, DType::I64, MemScope::Register, Dim::from(1));
            let off_idx = kernel.insert_const_idx_before(t.update, 0i64);
            // The wrap span of the K offset: the whole contraction in the
            // operand's K stride. The loop length is an expression, so the
            // span is one too.
            let (a_span, b_span) = {
                let sa = kernel.insert_const_idx_before(t.update, a_step * t.a_k);
                let a_span = kernel.insert_before(t.update, Op::Binary { x: loop_len_id, y: sa, bop: BOp::Mul });
                let sb = kernel.insert_const_idx_before(t.update, b_step * t.b_k);
                let b_span = kernel.insert_before(t.update, Op::Binary { x: loop_len_id, y: sb, bop: BOp::Mul });
                (a_span, b_span)
            };
            let a_inc = win_k * t.a_k;
            let b_inc = win_k * t.b_k;
            // A's fill of the iteration reads the element one chunk past the
            // one the instructions consume: `loop * a_step + win_k`, in A's K
            // stride.
            let mut a_init = {
                let st = kernel.insert_const_idx_before(t.update, a_step);
                kernel.insert_before(t.update, Op::Binary { x: t.loop_id, y: st, bop: BOp::Mul })
            };
            {
                let w = kernel.insert_const_idx_before(t.update, win_k);
                a_init = kernel.insert_before(t.update, Op::Binary { x: a_init, y: w, bop: BOp::Add });
            }
            if t.a_k != 1 {
                let s = kernel.insert_const_idx_before(t.update, t.a_k);
                a_init = kernel.insert_before(t.update, Op::Binary { x: a_init, y: s, bop: BOp::Mul });
            }
            kernel.insert_before(t.update, Op::Store { dst: a_off_st, dst_idx: off_idx, src: a_init });
            // The same for B, in B's K stride.
            let mut b_init = {
                let st = kernel.insert_const_idx_before(t.update, b_step);
                kernel.insert_before(t.update, Op::Binary { x: t.loop_id, y: st, bop: BOp::Mul })
            };
            {
                let w = kernel.insert_const_idx_before(t.update, win_k);
                b_init = kernel.insert_before(t.update, Op::Binary { x: b_init, y: w, bop: BOp::Add });
            }
            if t.b_k != 1 {
                let s = kernel.insert_const_idx_before(t.update, t.b_k);
                b_init = kernel.insert_before(t.update, Op::Binary { x: b_init, y: s, bop: BOp::Mul });
            }
            kernel.insert_before(t.update, Op::Store { dst: b_off_st, dst_idx: off_idx, src: b_init });
            // The warp's staged fragments: one section per `ldmatrix` lane
            // (`operand * pairs + pair`), each `mt * width` halves, indexed by
            // its tile lane so one `wmma` + one accumulator store cover the
            // whole tile grid.
            let steps_here = 2;
            // The body is the K loop's own: the chunk in shared memory is
            // consumed by the instructions, then overwritten by the chunk the
            // body loaded at its top.
            let end = t.update;
            {
                // The fill reads the carried element, and the next chunk's
                // element is this one advanced by a chunk and wrapped into the
                // contraction.
                let a_off = kernel.insert_before(end, Op::Load { src: a_off_st, src_idx: off_idx });
                let a_fill = kernel.insert_before(end, Op::Binary { x: a_origin, y: a_off, bop: BOp::Add });
                let b_off = kernel.insert_before(end, Op::Load { src: b_off_st, src_idx: off_idx });
                let b_fill = kernel.insert_before(end, Op::Binary { x: b_origin, y: b_off, bop: BOp::Add });
                let a_next = {
                    let inc = kernel.insert_const_idx_before(end, a_inc);
                    let next = kernel.insert_before(end, Op::Binary { x: a_off, y: inc, bop: BOp::Add });
                    let span = a_span;
                    let cond = kernel.insert_before(end, Op::Binary { x: next, y: span, bop: BOp::Cmpge });
                    let over = kernel.insert_before(end, Op::Binary { x: next, y: span, bop: BOp::Sub });
                    kernel.insert_before(end, Op::Select { cond, if_true: over, if_false: next })
                };
                let b_next = {
                    let inc = kernel.insert_const_idx_before(end, b_inc);
                    let next = kernel.insert_before(end, Op::Binary { x: b_off, y: inc, bop: BOp::Add });
                    let span = b_span;
                    let cond = kernel.insert_before(end, Op::Binary { x: next, y: span, bop: BOp::Cmpge });
                    let over = kernel.insert_before(end, Op::Binary { x: next, y: span, bop: BOp::Sub });
                    kernel.insert_before(end, Op::Select { cond, if_true: over, if_false: next })
                };
                kernel.insert_before(end, Op::Store { dst: a_off_st, dst_idx: off_idx, src: a_next });
                kernel.insert_before(end, Op::Store { dst: b_off_st, dst_idx: off_idx, src: b_next });
                // The next chunk's loads go out here, before the compute of
                // this one; their stores follow it, so the global round trip is
                // covered by the chunk's HMMAs.
                let (asl, av) =
                    coop_fill_load(kernel, end, fill_lane, t.a_src, a_fill, t.a_m, t.a_k, a_rows, win_k, fill_threads);
                let (bsl, bv) =
                    coop_fill_load(kernel, end, fill_lane, t.b_src, b_fill, t.b_n, t.b_k, b_rows, win_k, fill_threads);
                // One lane base for the whole tile: every `ldmatrix` of every
                // operand, pair and tile adds only its own offset to it — A's
                // loads and B's pair loads are the same 16-row, 2-K-step shape.
                let a_lb = ldmatrix_lane_base(kernel, end, lane, fm, if paired { fk * 2 } else { fk }, a_rows);
                // The whole tile grid as one lane: `tm`/`tn` are the lane's
                // quotient and remainder (a shift and a mask, the way the
                // reference derives its lane geometry — the lane counts from
                // zero, so these are the arithmetic `div`/`mod`, and a literal
                // `div.s64` is a *signed* division that ptxas lowers to a call
                // sequence). One accumulator read per chunk and one write after
                // its last K step: the steps chain through the instruction's
                // `c`, which accumulates in place.
                let grid = mt * nt;
                let tl = kernel.insert_before(end, Op::Unroll { axis: 0, len: u16::try_from(grid).expect("tile count") });
                debug_assert!(nt.is_power_of_two(), "tile-grid lane split: n tile count is not a power of two");
                let shift = kernel.insert_const_idx_before(end, Dim::from(nt.trailing_zeros() as i64));
                let tml = kernel.insert_before(end, Op::Binary { x: tl, y: shift, bop: BOp::BitShiftRight });
                let mask = kernel.insert_const_idx_before(end, Dim::from(nt as i64 - 1));
                let tnl = kernel.insert_before(end, Op::Binary { x: tl, y: mask, bop: BOp::BitAnd });
                let acc_idx = acc_frag_index(kernel, t, v, nt_w, tml, tnl, 1, end);
                // The K pairs are the only serial thing here — each advances
                // the same accumulator chain and its K offset is a constant per
                // load — so they are the one iteration emitted by a Rust loop:
                // every `ldmatrix` then carries its sub-tile as an immediate
                // (`+0x1000`, `+0x4000`, ...), as the reference's unrolled body
                // does. Everything else is a lane.
                let npairs = (win_steps / 2).max(1);
                let eight = Dim::from(8i64);
                // One `ldmatrix` for both operands and every K pair: the lane is
                // `operand * npairs + pair`, B sits one A-section later in the
                // same tile, and each pair is one k-block pair further in, so the
                // lane's slot is `lane * pair_stride` — `[Rbase]`,
                // `[Rbase+0x1000]`, ... the way the reference reads its single
                // 32 KB buffer. The tile is the next lane and the fragment width
                // the innermost, so one op covers all eight loads.
                debug_assert_eq!(a_rows, b_rows, "one tile for both operands needs one row count");
                debug_assert_eq!(mt, nt / 2, "one tile lane for both operands needs one tile count");
                let width = a_slots * if paired { 2 } else { 1 };
                let nlanes = 2 * npairs;
                let ab_st =
                    insert_scoped_storage(kernel, DType::F16, MemScope::Register, Dim::from((nlanes * mt * width) as i64));
                let obl =
                    kernel.insert_before(end, Op::Unroll { axis: 3, len: u16::try_from(nlanes).expect("operand-pair lane") });
                let tlane = kernel.insert_before(end, Op::Unroll { axis: 0, len: u16::try_from(mt).expect("tile count") });
                // The lane's slot: its pair section, its tile, and the warp's own
                // band (A's rows for A's lanes, B's for B's).
                let ps = kernel.insert_const_idx_before(end, Dim::from(2 * fk) * a_rows);
                let mut sb = kernel.insert_before(end, Op::Binary { x: obl, y: ps, bop: BOp::Mul });
                let ts = kernel.insert_const_idx_before(end, fm * eight);
                let toff = kernel.insert_before(end, Op::Binary { x: tlane, y: ts, bop: BOp::Mul });
                sb = kernel.insert_before(end, Op::Binary { x: sb, y: toff, bop: BOp::Add });
                if !warp_off_m.is_null() || !warp_off_n.is_null() {
                    let wa = if warp_off_m.is_null() {
                        kernel.insert_const_idx_before(end, 0i64)
                    } else {
                        let s = kernel.insert_const_idx_before(end, eight);
                        kernel.insert_before(end, Op::Binary { x: warp_off_m, y: s, bop: BOp::Mul })
                    };
                    let wb = if warp_off_n.is_null() {
                        kernel.insert_const_idx_before(end, 0i64)
                    } else {
                        let s = kernel.insert_const_idx_before(end, eight);
                        kernel.insert_before(end, Op::Binary { x: warp_off_n, y: s, bop: BOp::Mul })
                    };
                    let half = kernel.insert_const_idx_before(end, Dim::from(npairs as i64));
                    let is_b = kernel.insert_before(end, Op::Binary { x: obl, y: half, bop: BOp::Cmpge });
                    let w = kernel.insert_before(end, Op::Select { cond: is_b, if_true: wb, if_false: wa });
                    sb = kernel.insert_before(end, Op::Binary { x: sb, y: w, bop: BOp::Add });
                }
                {
                    let value = coop_ldmatrix(
                        kernel,
                        end,
                        lane,
                        ab_buf,
                        fm,
                        if paired { fk * 2 } else { fk },
                        a_rows,
                        Dim::from(0),
                        // The tile's k-block count: the read key is the k-block
                        // *inside* the tile, so B's base (a whole tile further
                        // in) must not shift it.
                        win_k / 8,
                        sb,
                        a_lb,
                    );
                    // `[lane][tile][half]`: one contiguous run per tile, so one
                    // store per lane writes the whole `ldmatrix` result.
                    let sect = kernel.insert_const_idx_before(end, Dim::from((mt * width) as i64));
                    let base = kernel.insert_before(end, Op::Binary { x: obl, y: sect, bop: BOp::Mul });
                    let ws = kernel.insert_const_idx_before(end, Dim::from(width as i64));
                    let tb = kernel.insert_before(end, Op::Binary { x: tlane, y: ws, bop: BOp::Mul });
                    let base = kernel.insert_before(end, Op::Binary { x: base, y: tb, bop: BOp::Add });
                    let wl =
                        kernel.insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(width).expect("fragment width") });
                    let idx = kernel.insert_before(end, Op::Binary { x: base, y: wl, bop: BOp::Add });
                    kernel.insert_before(end, Op::Store { dst: ab_st, dst_idx: idx, src: value });
                }
                // The K pairs are serial — each step's `c` is the previous
                // result — so every pair reads its own staged section by a
                // constant and only the chain is a loop. The accumulator itself
                // is read once for the whole chunk and written back once: the
                // register storage is the K loop's carried state, not a hop per
                // pair, so the `wmma` chain stays live in registers across all
                // of them the way the reference's 64 accumulators do.
                let mut acc = kernel.insert_before(end, Op::Load { src: t.acc, src_idx: acc_idx });
                for pi in 0..npairs {
                    // One `wmma` per (tile, K step) covers the grid: `a`/`b` are
                    // lane-indexed reads of the staged sections and the chain
                    // runs through the instruction's `c`.
                    for sl in 0..steps_here {
                        // A: `[operand][pair][tile][step][slot]`, operand 0.
                        let asec = kernel.insert_const_idx_before(end, Dim::from((pi * mt * width) as i64));
                        let ws = kernel.insert_const_idx_before(end, Dim::from(width as i64));
                        let abase = kernel.insert_before(end, Op::Binary { x: tml, y: ws, bop: BOp::Mul });
                        let abase = kernel.insert_before(end, Op::Binary { x: abase, y: asec, bop: BOp::Add });
                        let am = kernel.insert_const_idx_before(end, Dim::from((sl * a_slots) as i64));
                        let abase = kernel.insert_before(end, Op::Binary { x: abase, y: am, bop: BOp::Add });
                        let alane = kernel
                            .insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(a_slots).expect("a fragment slots") });
                        let aidx = kernel.insert_before(end, Op::Binary { x: abase, y: alane, bop: BOp::Add });
                        let a_frag = kernel.insert_before(end, Op::Load { src: ab_st, src_idx: aidx });
                        // B: operand 1, the `ldmatrix`'s own word order:
                        // `(tile-in-pair) + 2 * K step`, one `b_slots` each.
                        let bsec = kernel.insert_const_idx_before(end, Dim::from(((npairs + pi) * mt * width) as i64));
                        let two = kernel.insert_const_idx_before(end, Dim::from(2i64));
                        let tpair = kernel.insert_before(end, Op::Binary { x: tnl, y: two, bop: BOp::Div });
                        let bbase = kernel.insert_before(end, Op::Binary { x: tpair, y: ws, bop: BOp::Mul });
                        let bbase = kernel.insert_before(end, Op::Binary { x: bbase, y: bsec, bop: BOp::Add });
                        let trem = kernel.insert_before(end, Op::Binary { x: tnl, y: two, bop: BOp::Mod });
                        let tmn = kernel.insert_const_idx_before(end, Dim::from(2 * sl as i64));
                        let tidx = kernel.insert_before(end, Op::Binary { x: trem, y: tmn, bop: BOp::Add });
                        let bsl = kernel.insert_const_idx_before(end, Dim::from(b_slots as i64));
                        let bmid = kernel.insert_before(end, Op::Binary { x: tidx, y: bsl, bop: BOp::Mul });
                        let bbase = kernel.insert_before(end, Op::Binary { x: bbase, y: bmid, bop: BOp::Add });
                        let blane = kernel
                            .insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(b_slots).expect("b fragment slots") });
                        let bidx = kernel.insert_before(end, Op::Binary { x: bbase, y: blane, bop: BOp::Add });
                        let b_frag = kernel.insert_before(end, Op::Load { src: ab_st, src_idx: bidx });
                        acc = kernel.insert_before(
                            end,
                            Op::Wmma { dims: v.dims, layout: v.layout, dtype: v.dtype, a: a_frag, b: b_frag, c: acc },
                        );
                    }
                }
                kernel.insert_before(end, Op::Store { dst: t.acc, dst_idx: acc_idx, src: acc });
                // Read-protect: no thread may overwrite the tile until every
                // warp has finished the `ldmatrix`es of the chunk in it. Then
                // the next chunk lands in the same 16 KB, and the second
                // barrier publishes it for the next iteration's `ldmatrix`es.
                kernel.insert_before(end, Op::Barrier);
                coop_fill_store(kernel, end, ab_buf, OpId::NULL, asl, av);
                coop_fill_store(kernel, end, ab_buf, a_win_c, bsl, bv);
                kernel.insert_before(end, Op::Barrier);
            }
        }
    } else {
        for tm in 0..mt_w {
            for tn in 0..nt_w {
                let tmo = kernel.insert_const_idx_before(t.update, tm);
                let tno = kernel.insert_const_idx_before(t.update, tn);
                // One accumulator read per (tile, K loop), one write after the
                // last K step: the steps chain through the instruction's `c`.
                let acc_idx = acc_frag_index(kernel, t, v, nt_w, tmo, tno, 0, t.update);
                let mut acc = kernel.insert_before(t.update, Op::Load { src: t.acc, src_idx: acc_idx });
                for tk in 0..t.kt {
                    let a_base = base_expr(kernel, t.update, &a_fixed, t.a_offset, tm * fm * t.a_m + tk * fk * t.a_k);
                    let a_base = shift(kernel, t.update, a_base, [warp_off_a, a_band]);
                    let b_base = base_expr(kernel, t.update, &b_fixed, t.b_offset, tn * fn_ * t.b_n + tk * fk * t.b_k);
                    let b_base = shift(kernel, t.update, b_base, [warp_off_b, b_band]);
                    let a_elems = frag_elems(kernel, t.update, lane, &v.a, t.a_src, a_base, t.a_m, t.a_k, warp);
                    let a_frag = kernel.insert_before(t.update, Op::Stack { ops: a_elems.into() });
                    let b_elems = frag_elems(kernel, t.update, lane, &v.b, t.b_src, b_base, t.b_n, t.b_k, warp);
                    let b_frag = kernel.insert_before(t.update, Op::Stack { ops: b_elems.into() });
                    acc = kernel.insert_before(
                        t.update,
                        Op::Wmma { dims: v.dims, layout: v.layout, dtype: v.dtype, a: a_frag, b: b_frag, c: acc },
                    );
                }
                kernel.insert_before(t.update, Op::Store { dst: t.acc, dst_idx: acc_idx, src: acc });
            }
        }
    }

    // The op that follows the K loop. The sum below has to run after the loop
    // and before every store that reads the accumulator, so it anchors on IR
    // order, not on the (unordered) epilogue list.
    let mut anchor = kernel.next_op(t.loop_id);
    let mut depth = 1u32;
    while !anchor.is_null() && depth > 0 {
        match kernel.at(anchor) {
            Op::Loop { .. } => depth += 1,
            Op::EndLoop => depth -= 1,
            _ => {}
        }
        anchor = kernel.next_op(anchor);
    }

    if combine && !staged && !anchor.is_null() {
        // Sum the per-warp partial accumulators: each warp writes its partial
        // tile to shared memory indexed by warp, then every warp re-reads and
        // adds all partials.
        let acc_len = t.mt * t.nt * t.slots;
        let red = insert_local_storage(kernel, DType::F32, Dim::from(wk) * Dim::from(warp) * acc_len);
        let cw = |kernel: &mut Kernel| -> OpId { kernel.insert_const_idx_before(anchor, Dim::from(warp)) };
        let cl = |kernel: &mut Kernel| -> OpId { kernel.insert_const_idx_before(anchor, acc_len) };
        // One lane over the accumulator's elements: the partial write and the
        // cross-warp sum are each one op, duplicated per element.
        let il =
            kernel.insert_before(anchor, Op::Unroll { axis: 0, len: u16::try_from(acc_len).expect("accumulator element count") });
        {
            let at = il;
            let v = kernel.insert_before(anchor, Op::Load { src: t.acc, src_idx: at });
            let w = cw(kernel);
            let wl = kernel.insert_before(anchor, Op::Binary { x: warp_var, y: w, bop: BOp::Mul });
            let wl = kernel.insert_before(anchor, Op::Binary { x: wl, y: lane, bop: BOp::Add });
            let c = cl(kernel);
            let b = kernel.insert_before(anchor, Op::Binary { x: wl, y: c, bop: BOp::Mul });
            let idx = kernel.insert_before(anchor, Op::Binary { x: b, y: at, bop: BOp::Add });
            kernel.insert_before(anchor, Op::Store { dst: red, dst_idx: idx, src: v });
        }
        kernel.insert_before(anchor, Op::Barrier);
        {
            let mut sum = OpId::NULL;
            for w in 0..wk {
                let cw = kernel.insert_const_idx_before(anchor, Dim::from(w) * Dim::from(warp));
                let wl = kernel.insert_before(anchor, Op::Binary { x: cw, y: lane, bop: BOp::Add });
                let c = cl(kernel);
                let b = kernel.insert_before(anchor, Op::Binary { x: wl, y: c, bop: BOp::Mul });
                let idx = kernel.insert_before(anchor, Op::Binary { x: b, y: il, bop: BOp::Add });
                let v = kernel.insert_before(anchor, Op::Load { src: red, src_idx: idx });
                sum = if sum.is_null() {
                    v
                } else {
                    kernel.insert_before(anchor, Op::Binary { x: sum, y: v, bop: BOp::Add })
                };
            }
            kernel.insert_before(anchor, Op::Store { dst: t.acc, dst_idx: il, src: sum });
        }
    }

    // llama.cpp `mul_mat_f`'s epilogue: every warp holds the whole tile, so
    // each writes its partial into a shared buffer laid out
    // `[output column][warp * rows + row]`; after the barrier the warps take
    // turns owning one output column each, sum the per-warp copies and store
    // one lane per output row.
    if staged && combine && !anchor.is_null() && !t.epilogue.is_empty() {
        let rows = t.mt * fm;
        let cols = t.nt * fn_;
        let kiw = Dim::from(wk) * cols + Dim::from(4);
        let combine_buf = insert_local_storage(kernel, DType::F32, rows * kiw);
        // llama.cpp always runs with `threadIdx.y`; one warp is warp 0.
        let wid = if warp_var.is_null() {
            kernel.insert_const_idx_before(anchor, 0i64)
        } else {
            warp_var
        };
        let crow = lane_coord(kernel, anchor, lane, &v.c, FragAxis::Row, warp);
        let ccol = lane_coord(kernel, anchor, lane, &v.c, FragAxis::Col, warp);
        for tm in 0..t.mt {
            for tn in 0..t.nt {
                for (s, slot) in v.c.slots.iter().enumerate() {
                    let moff = kernel.insert_const_idx_before(anchor, tm * fm + Dim::from(slot.0));
                    let mrow = kernel.insert_before(anchor, Op::Binary { x: moff, y: crow, bop: BOp::Add });
                    let ck = kernel.insert_const_idx_before(anchor, kiw);
                    let mbase = kernel.insert_before(anchor, Op::Binary { x: mrow, y: ck, bop: BOp::Mul });
                    let wmul = kernel.insert_const_idx_before(anchor, cols);
                    let wbase = kernel.insert_before(anchor, Op::Binary { x: wid, y: wmul, bop: BOp::Mul });
                    let c0 = kernel.insert_before(anchor, Op::Binary { x: wbase, y: ccol, bop: BOp::Add });
                    let c1 = kernel.insert_const_idx_before(anchor, tn * fn_ + Dim::from(slot.1));
                    let cidx = kernel.insert_before(anchor, Op::Binary { x: c0, y: c1, bop: BOp::Add });
                    let idx = kernel.insert_before(anchor, Op::Binary { x: mbase, y: cidx, bop: BOp::Add });
                    let at = acc_index(kernel, anchor, nt_w, t.slots, tm, tn, s);
                    let val = kernel.insert_before(anchor, Op::Load { src: t.acc, src_idx: at });
                    kernel.insert_before(anchor, Op::Store { dst: combine_buf, dst_idx: idx, src: val });
                }
            }
        }
        // The combine buffer is read by warps other than the writer's with
        // more than one warp; llama.cpp syncs the warp otherwise.
        if wk > 1 {
            kernel.insert_before(anchor, Op::Barrier);
        } else {
            kernel.insert_before(anchor, Op::PTX(Box::new(PTXOp::WarpSync)));
        }
        let rounds = usize::try_from(rows).expect("llama epilogue: rows") / usize::from(wk);
        for epi in &t.epilogue {
            for r in 0..rounds {
                let moff = kernel.insert_const_idx_before(anchor, r as Dim * Dim::from(wk));
                let m = kernel.insert_before(anchor, Op::Binary { x: wid, y: moff, bop: BOp::Add });
                let ck = kernel.insert_const_idx_before(anchor, kiw);
                let mbase = kernel.insert_before(anchor, Op::Binary { x: m, y: ck, bop: BOp::Mul });
                let mut sum = OpId::NULL;
                for w in 0..wk {
                    let woff = kernel.insert_const_idx_before(anchor, w as Dim * cols);
                    let wl = kernel.insert_before(anchor, Op::Binary { x: woff, y: lane, bop: BOp::Add });
                    let idx = kernel.insert_before(anchor, Op::Binary { x: mbase, y: wl, bop: BOp::Add });
                    let val = kernel.insert_before(anchor, Op::Load { src: combine_buf, src_idx: idx });
                    sum = if sum.is_null() {
                        val
                    } else {
                        kernel.insert_before(anchor, Op::Binary { x: sum, y: val, bop: BOp::Add })
                    };
                }
                let mut elem = sum;
                for dtype in epi.chain.iter().rev() {
                    elem = kernel.insert_before(anchor, Op::Cast { x: elem, dtype: *dtype });
                }
                let mut addr = base_expr(kernel, anchor, &epi.fixed, epi.offset, 0);
                let mm = if epi.m == 1 {
                    m
                } else {
                    let s = kernel.insert_const_idx_before(anchor, epi.m);
                    kernel.insert_before(anchor, Op::Binary { x: m, y: s, bop: BOp::Mul })
                };
                addr = kernel.insert_before(anchor, Op::Binary { x: addr, y: mm, bop: BOp::Add });
                let ll = if epi.n == 1 {
                    lane
                } else {
                    let s = kernel.insert_const_idx_before(anchor, epi.n);
                    kernel.insert_before(anchor, Op::Binary { x: lane, y: s, bop: BOp::Mul })
                };
                addr = kernel.insert_before(anchor, Op::Binary { x: addr, y: ll, bop: BOp::Add });
                kernel.insert_before(anchor, Op::Store { dst: epi.dst, dst_idx: addr, src: elem });
            }
        }
    }

    // Output stores: the accumulator keeps the values, each slot is written
    // at its own output coordinate. The warp's tile grid is two lanes
    // (`mt_w` x `nt_w` copies), so the store is written once here and
    // `devectorize` duplicates it per tile — the lane is an index vector, so
    // the full address runs on it unchanged.
    for epi in &t.epilogue {
        if staged && combine {
            break;
        }
        // The warp's tile grid as one lane per axis: the store is written once
        // and `devectorize` duplicates it per tile.
        let tm = if mt_w > 1 {
            kernel.insert_before(epi.store, Op::Unroll { axis: 0, len: u16::try_from(mt_w).expect("m tile count") })
        } else {
            kernel.insert_const_idx_before(epi.store, 0i64)
        };
        let tn = if nt_w > 1 {
            kernel.insert_before(epi.store, Op::Unroll { axis: 1, len: u16::try_from(nt_w).expect("n tile count") })
        } else {
            kernel.insert_const_idx_before(epi.store, 0i64)
        };
        {
            // The value of one accumulator slot, cast through the
            // epilogue's dtype chain.
            // The instruction slots are a lane too: the accumulator
            // read, the cast chain, the address and the store are one op
            // each, over three lanes — the warp's tile grid and the slots.
            let nslots = v.c.slots.len();
            let half = u16::try_from(nslots / 2).expect("fragment slot rows");
            let drow = Dim::from(v.c.slots.get(half as usize).map(|s| s.0).unwrap_or(0) as i64);
            let dcol = Dim::from(v.c.slots.get(1).map(|s| s.1).unwrap_or(0) as i64);
            let sl = kernel.insert_before(epi.store, Op::Unroll { axis: 2, len: u16::try_from(nslots).expect("fragment slots") });
            let hc = kernel.insert_const_idx_before(epi.store, Dim::from(half as i64));
            let slot_row = kernel.insert_before(epi.store, Op::Binary { x: sl, y: hc, bop: BOp::Div });
            let slot_col = kernel.insert_before(epi.store, Op::Binary { x: sl, y: hc, bop: BOp::Mod });
            {
                // Value: the whole fragment, cast element-wise.
                let ntc = kernel.insert_const_idx_before(epi.store, nt_w);
                let mut idx = kernel.insert_before(epi.store, Op::Binary { x: tm, y: ntc, bop: BOp::Mul });
                idx = kernel.insert_before(epi.store, Op::Binary { x: idx, y: tn, bop: BOp::Add });
                let slots_c = kernel.insert_const_idx_before(epi.store, t.slots);
                idx = kernel.insert_before(epi.store, Op::Binary { x: idx, y: slots_c, bop: BOp::Mul });
                idx = kernel.insert_before(epi.store, Op::Binary { x: idx, y: sl, bop: BOp::Add });
                let mut elem = kernel.insert_before(epi.store, Op::Load { src: t.acc, src_idx: idx });
                for dtype in epi.chain.iter().rev() {
                    elem = kernel.insert_before(epi.store, Op::Cast { x: elem, dtype: *dtype });
                }
                // Address: `base + (tm*fm + slot_row*drow)*m + (tn*fn +
                // slot_col*dcol)*n`, plus the warp's window and lane.
                let mut addr = base_expr(kernel, epi.store, &epi.fixed, epi.offset, 0);
                let fmc = kernel.insert_const_idx_before(epi.store, fm);
                let mut mrow = kernel.insert_before(epi.store, Op::Binary { x: tm, y: fmc, bop: BOp::Mul });
                let dc = kernel.insert_const_idx_before(epi.store, drow);
                let moff = kernel.insert_before(epi.store, Op::Binary { x: slot_row, y: dc, bop: BOp::Mul });
                mrow = kernel.insert_before(epi.store, Op::Binary { x: mrow, y: moff, bop: BOp::Add });
                let mut extra = if epi.m != 1 {
                    let st = kernel.insert_const_idx_before(epi.store, epi.m);
                    kernel.insert_before(epi.store, Op::Binary { x: mrow, y: st, bop: BOp::Mul })
                } else {
                    mrow
                };
                let fnc = kernel.insert_const_idx_before(epi.store, fn_);
                let mut ncol = kernel.insert_before(epi.store, Op::Binary { x: tn, y: fnc, bop: BOp::Mul });
                let dc2 = kernel.insert_const_idx_before(epi.store, dcol);
                let noff = kernel.insert_before(epi.store, Op::Binary { x: slot_col, y: dc2, bop: BOp::Mul });
                ncol = kernel.insert_before(epi.store, Op::Binary { x: ncol, y: noff, bop: BOp::Add });
                let nterm = if epi.n != 1 {
                    let st = kernel.insert_const_idx_before(epi.store, epi.n);
                    kernel.insert_before(epi.store, Op::Binary { x: ncol, y: st, bop: BOp::Mul })
                } else {
                    ncol
                };
                extra = kernel.insert_before(epi.store, Op::Binary { x: extra, y: nterm, bop: BOp::Add });
                addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: extra, bop: BOp::Add });
                for (off, stride) in [(warp_off_m, epi.m), (warp_off_n, epi.n)] {
                    if off.is_null() {
                        continue;
                    }
                    let st = kernel.insert_const_idx_before(epi.store, stride);
                    let t2 = kernel.insert_before(epi.store, Op::Binary { x: off, y: st, bop: BOp::Mul });
                    addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: t2, bop: BOp::Add });
                }
                let row = lane_coord(kernel, epi.store, lane, &v.c, FragAxis::Row, warp);
                let rowt = if epi.m != 1 {
                    let st = kernel.insert_const_idx_before(epi.store, epi.m);
                    kernel.insert_before(epi.store, Op::Binary { x: row, y: st, bop: BOp::Mul })
                } else {
                    row
                };
                addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: rowt, bop: BOp::Add });
                let col = lane_coord(kernel, epi.store, lane, &v.c, FragAxis::Col, warp);
                let colt = if epi.n != 1 {
                    let st = kernel.insert_const_idx_before(epi.store, epi.n);
                    kernel.insert_before(epi.store, Op::Binary { x: col, y: st, bop: BOp::Mul })
                } else {
                    col
                };
                addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: colt, bop: BOp::Add });
                kernel.insert_before(epi.store, Op::Store { dst: epi.dst, dst_idx: addr, src: elem });
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
    /// Make the tensorization optimization: one config per candidate in the
    /// whole tensorization space — every scalar matmul region the IR already
    /// has, every shape whose lanes a split already built, and every tile grid
    /// the device's instructions allow, each at every warp layout and in both
    /// operand forms (staged through shared memory, or read from global).
    /// Never mutates.
    pub fn opt_tensorize(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_tensorize");
        Box::new(self.tensorize_impl(WARP_LAYOUTS.to_vec()))
    }

    /// Read-only: the matmul forms this kernel already has, as it stands —
    /// the scalar regions the region matcher finds and the pre-tiled shapes a
    /// split has already built. Never mutates and never clones.
    fn tensor_forms(&self) -> (Vec<Region>, Vec<Tiled>) {
        let mut regions = Vec::new();
        let info = self.device_info();
        // Device gate: a tensor-core unit offering at least one instruction
        // variant.
        if !info.tensor_cores || info.mma_variants.is_empty() {
            return (regions, Vec::new());
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
        // Consumers of every op, for the accumulator gate below.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        for &id in &order {
            for p in self.at(id).parameters().filter(|p| !p.is_null()) {
                users.entry(p).or_default().push(id);
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
            // Every store that reads the accumulator through a cast chain is
            // one C output this region rewrites. Collect them here so `apply`
            // never rescans the IR. Refuse the region when the accumulator has
            // any other reader: a fused `silu(acc)` (the swiglu kernel)
            // consumes the scalar value outside that chain, and `apply` would
            // leave it reading a storage whose stores it deletes.
            let mut epilogue: Vec<(OpId, OpId, Vec<crate::DType>)> = Vec::new();
            let mut epi_loads: Vec<OpId> = Vec::new();
            for &id in &order {
                let Op::Store { dst, src, .. } = self.at(id) else {
                    continue;
                };
                let mut chain: Vec<crate::DType> = Vec::new();
                let mut val = *src;
                loop {
                    match self.at(val) {
                        Op::Cast { x, dtype } | Op::Bitcast { x, dtype } => {
                            chain.push(*dtype);
                            val = *x;
                        }
                        Op::Load { src: s, .. } if *s == acc => {
                            epilogue.push((id, *dst, chain));
                            if !epi_loads.contains(&val) {
                                epi_loads.push(val);
                            }
                            break;
                        }
                        _ => break,
                    }
                }
            }
            if epilogue.is_empty() {
                continue;
            }
            // The epilogue stores may share one accumulator load (the llama
            // seed stores `f16(a)` and `f32(f16(a))`), so count distinct loads.
            let n_loads = order.iter().filter(|&&id| matches!(self.at(id), Op::Load { src, .. } if *src == acc)).count();
            if n_loads != epi_loads.len() + 1 {
                continue;
            }
            // Every consumer reachable from an epilogue load must be a cast or
            // one of the epilogue stores; anything else (e.g. `silu(acc)`)
            // still needs the scalar value after the stores are gone.
            let stores: Set<OpId> = epilogue.iter().map(|(store, _, _)| *store).collect();
            let mut store_only = true;
            for &load in &epi_loads {
                let mut stack = users.get(&load).cloned().unwrap_or_default();
                let mut seen: Set<OpId> = Set::default();
                while let Some(u) = stack.pop() {
                    if !seen.insert(u) {
                        continue;
                    }
                    if stores.contains(&u) {
                        continue;
                    }
                    if matches!(self.at(u), Op::Cast { .. } | Op::Bitcast { .. }) {
                        stack.extend(users.get(&u).cloned().unwrap_or_default());
                    } else {
                        store_only = false;
                        break;
                    }
                }
                if !store_only {
                    break;
                }
            }
            if !store_only {
                continue;
            }
            let &Op::Store { dst_idx: c_idx, .. } = self.at(final_store) else {
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
                    update,
                    epilogue: epilogue.clone(),
                    a_src,
                    a_m: a_map[&m_range].1,
                    a_k: a_map[&k_loop].1,
                    b_src,
                    b_n: b_map[&n_range].1,
                    b_k: b_map[&k_loop].1,
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
                let Some(&(k_var, k_len_u16)) = a_unrolls.iter().find(|(v, _)| b_unrolls.iter().any(|(w, _)| w == v)) else {
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
                    // A wider K step unrolls the loop body instead of adding
                    // iterations, so the ceiling is the emitted body: the
                    // staged form's is one loop over its fragments whatever
                    // the tile grid, the unstage form's is the grid itself
                    // (bounded where that body is emitted).
                    if mt.saturating_mul(nt).saturating_mul(kt) > 2048 {
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
                        slots: Dim::from(variant.c.slots.len() as i64),
                        init,
                        loop_id,
                        update,
                        epilogue: epilogue.clone(),
                    });
                }
            }
        }
        (regions, tiled)
    }

    /// Make the tensorization optimization. Enumeration only: it reads the
    /// kernel, never clones it and never applies anything.
    fn tensorize_impl(&self, layouts: Vec<(u16, u16, u16)>) -> Tensorize {
        let (regions, tiled) = self.tensor_forms();
        let info = self.device_info();
        let mut plans: Vec<Plan> = Vec::new();
        // One plan per matched shape, at every warp layout, in both operand
        // forms. A register (unstage) block sums a K split through
        // `mt * nt * slots` registers per warp, an IR and register blowout, so
        // only the staged form offers a split-K layout.
        for i in 0..(regions.len() + tiled.len()) {
            let (region, tiled_idx) = if i < regions.len() {
                (Some(i), None)
            } else {
                (None, Some(i - regions.len()))
            };
            for &layout in &layouts {
                for staged in [false, true] {
                    if !staged && layout.2 > 1 {
                        continue;
                    }
                    plans.push(Plan { region, tiled: tiled_idx, grid: None, layout, staged });
                }
            }
        }

        // Tile-grid plans: for a kernel that still has the scalar matmul loop
        // and no lane split, barrier, reduce or shared tile from another pass,
        // this pass may build the grid itself. Every grid the device's
        // instructions allow over provably divisible extents.
        let plain = self.iter_unordered().all(|(_, op)| {
            !matches!(
                op,
                Op::Unroll { .. }
                    | Op::Barrier
                    | Op::Reduce { .. }
                    | Op::Wmma { .. }
                    | Op::Storage { scope: MemScope::Local, .. }
                    | Op::Range { kind: RangeKind::Local(_), .. }
            )
        });
        if plain {
            /// Lane multipliers on the instruction tile.
            const MULT: &[u16] = &[32, 16, 8, 4, 2, 1];
            /// The tile grids one plan may emit: the staged body is one loop
            /// over its fragments whatever the tile count, and a grid beyond
            /// this is not a candidate the search samples.
            const MAX_TILES: Dim = 2048;

            // The operand element dtype behind a value: a param, a storage, or
            // a load of either.
            let elem_dtype = |k: &Kernel, o: OpId| -> Option<crate::DType> {
                match k.at(o) {
                    Op::Param { dtype, .. } | Op::Storage { dtype, .. } => Some(*dtype),
                    Op::Load { src, .. } => match k.at(*src) {
                        Op::Param { dtype, .. } | Op::Storage { dtype, .. } => Some(*dtype),
                        _ => None,
                    },
                    _ => None,
                }
            };
            let mut ranges: Vec<(OpId, u32, OpId)> = Vec::new();
            let mut loop_id = OpId::NULL;
            let mut loop_len = OpId::NULL;
            let mut operand_dtypes: Option<(crate::DType, crate::DType)> = None;
            let mut acc_dtype: Option<crate::DType> = None;
            let mut op_id = self.head;
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Range { axis, kind: RangeKind::Group(len) } => {
                        if !ranges.iter().any(|r| r.1 == axis) {
                            ranges.push((op_id, axis, len));
                        }
                    }
                    Op::Loop { len } if loop_id.is_null() => {
                        loop_id = op_id;
                        loop_len = len;
                    }
                    Op::Binary { bop: BOp::Mul, x, y } if operand_dtypes.is_none() => {
                        operand_dtypes = elem_dtype(self, x).zip(elem_dtype(self, y));
                    }
                    Op::Storage { scope: MemScope::Register, dtype, .. } => acc_dtype = Some(dtype),
                    _ => {}
                }
                op_id = self.next_op(op_id);
            }
            if ranges.len() == 2
                && !loop_id.is_null()
                && !loop_len.is_null()
                && let (Some((x_dtype, y_dtype)), Some(acc_dtype)) = (operand_dtypes, acc_dtype)
            {
                ranges.sort_by_key(|&(_, axis, _)| axis);
                let ((m_range, _, m), (n_range, _, n)) = (ranges[0], ranges[1]);
                for variant in &info.mma_variants {
                    // The orientation is decided by the accumulator's view, not
                    // by the multiply order, so either operand may be A.
                    let dtypes_match = (variant.a_dtype == x_dtype && variant.b_dtype == y_dtype
                        || variant.a_dtype == y_dtype && variant.b_dtype == x_dtype)
                        && variant.c_dtype == acc_dtype;
                    if !dtypes_match {
                        continue;
                    }
                    let (fm, fn_, fk) = {
                        let (fm, fn_, fk) = variant.dims.decompose_mnk();
                        (fm as Dim, fn_ as Dim, fk as Dim)
                    };
                    for &mm in MULT {
                        let m_lanes = fm * Dim::from(mm);
                        if !self.is_sym_multiple(m, m_lanes) {
                            continue;
                        }
                        for &nn in MULT {
                            let n_lanes = fn_ * Dim::from(nn);
                            if !self.is_sym_multiple(n, n_lanes) {
                                continue;
                            }
                            for &kk in MULT {
                                let k_lanes = fk * Dim::from(kk);
                                if !self.is_sym_multiple(loop_len, k_lanes) {
                                    continue;
                                }
                                let tiles = (m_lanes / fm) * (n_lanes / fn_) * (k_lanes / fk);
                                if tiles > MAX_TILES {
                                    continue;
                                }
                                let (Ok(m_lanes), Ok(n_lanes), Ok(k_lanes)) =
                                    (u16::try_from(m_lanes), u16::try_from(n_lanes), u16::try_from(k_lanes))
                                else {
                                    continue;
                                };
                                for &layout in &layouts {
                                    for staged in [false, true] {
                                        if !staged && layout.2 > 1 {
                                            continue;
                                        }
                                        plans.push(Plan {
                                            region: None,
                                            tiled: None,
                                            grid: Some((m_range, n_range, loop_id, m_lanes, n_lanes, k_lanes, fm, fn_, fk)),
                                            layout,
                                            staged,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensorize { plans, regions, tiled }
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

    /// The config of the merged tensorization space for one shape (`0` for the
    /// first matched region or tile grid) at `layout`, in the staged or the
    /// unstage form. Plans are enumerated shape × layout × [unstage, staged],
    /// and a K-split layout is offered staged only, so it has one config.
    fn plan_config(shape: usize, layout: usize, staged: bool) -> u64 {
        let offered = |l: (u16, u16, u16)| if l.2 > 1 { 1 } else { 2 };
        let per_shape: usize = WARP_LAYOUTS.iter().map(|&l| offered(l)).sum();
        let before: usize = WARP_LAYOUTS[..layout].iter().map(|&l| offered(l)).sum();
        assert!(WARP_LAYOUTS[layout].2 == 1 || staged, "a K-split layout has no unstage form");
        let staged_offset = usize::from(staged) * usize::from(offered(WARP_LAYOUTS[layout]) == 2);
        (shape * per_shape + before + staged_offset) as u64
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
        assert_eq!(opt.nconfigs(), 46, "one region and one tile grid, per layout and operand form");
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
        assert_eq!(opt.nconfigs(), 46, "one NT region and one tile grid, per layout and operand form");
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
        assert!(opt.nconfigs() >= 23, "the llama prefill seed selects one region, per layout and operand form");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 1, "one wmma update");

        // Register block: `split` on the narrowed tile ranges, then the
        // epilogue's devectorize unfolds the lanes into one wmma per tile.
        k.split(m, split::SplitMode::Unroll, 4);
        k.split(n, split::SplitMode::Unroll, 4);
        k.default_epilogue();
        k.devectorize().expect("tensorized kernel devectorizes");
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
        assert_eq!(opt.nconfigs(), 23, "the tiled form matches one region, per warp layout and operand form");
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 32, "4 m tiles x 4 n tiles x 2 k steps");
        k.dead_code_elimination();
        let reduces = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).count();
        assert_eq!(reduces, 0, "the reduce is replaced");

        // Speed of the tiled match on the prefill projection.
        k.default_epilogue();
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
        assert!(k.iter_unordered().any(|(_, op)| matches!(op, Op::Stack { .. })), "the tensorized kernel has fragment stacks");

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
        // The staged form needs the block's output rows warp-wide (the
        // combine buffer stores one warp of rows per column).
        const N: i64 = 32;
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
        assert!(!k.split(n, split::SplitMode::Unroll, 32).is_null(), "n split");
        assert!(!k.split(k_loop, split::SplitMode::Unroll, 16).is_null(), "k split");
        k.verify();
        let scalar = k.clone();
        let opt = k.opt_tensorize();
        assert_eq!(opt.nconfigs(), 23, "the tiled form matches, per warp layout and operand form");
        opt.apply(&mut k, 0);
        k.default_epilogue();
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

        // The staged half: the cooperative shared tile + `ldmatrix` must
        // reproduce the same result as the direct fragment loads.
        let (mut ks, ms, ns, kls) = build();
        assert!(!ks.split(ms, split::SplitMode::Unroll, 32).is_null(), "staged m split");
        assert!(!ks.split(ns, split::SplitMode::Unroll, 32).is_null(), "staged n split");
        assert!(!ks.split(kls, split::SplitMode::Unroll, 16).is_null(), "staged k split");
        ks.verify();
        let sopt = ks.opt_tensorize();
        sopt.apply(&mut ks, plan_config(0, 0, true));
        ks.default_epilogue();
        ks.devectorize().expect("staged tensorized kernel devectorizes");
        ks.verify();
        assert!(
            ks.iter_unordered().any(|(_, op)| matches!(op, Op::PTX(_))),
            "the staged kernel reads its operands with ldmatrix"
        );
        let Some(staged) = run(ks, &a, &b) else { return };
        let mut staged_worst = 0.0f32;
        for i in 0..(M * N) as usize {
            staged_worst = staged_worst.max((staged[i] - want[i]).abs());
        }
        println!("staged vs scalar: max |err| = {staged_worst}");
        assert!(staged_worst < 1e-2, "staged tensorize disagrees with the scalar kernel: {staged_worst}");

        // Split-K across two warps: each warp runs one slice of the K loop and
        // the per-warp partial accumulators are summed through the combine
        // buffer. That is a shared-memory form, so the space offers it staged;
        // the multiple M and N tiles here check that the reduction and the
        // per-warp fragment bases hold for every tile.
        for staged in [true] {
            let (mut kw, mw, nw_, klw) = build();
            assert!(!kw.split(mw, split::SplitMode::Unroll, 32).is_null(), "split-K m split");
            assert!(!kw.split(nw_, split::SplitMode::Unroll, 32).is_null(), "split-K n split");
            assert!(!kw.split(klw, split::SplitMode::Unroll, 16).is_null(), "split-K k split");
            kw.verify();
            let wopt = kw.opt_tensorize();
            wopt.apply(&mut kw, plan_config(0, 10, staged));
            kw.default_epilogue();
            kw.verify();
            let Some(warped) = run(kw, &a, &b) else { return };
            let mut warped_worst = 0.0f32;
            for i in 0..(M * N) as usize {
                warped_worst = warped_worst.max((warped[i] - want[i]).abs());
            }
            println!("split-k (staged={staged}) vs scalar: max |err| = {warped_worst}");
            assert!(warped_worst < 1e-2, "split-K tensorize (staged={staged}) disagrees with the scalar kernel: {warped_worst}");
        }
    }

    /// The staged tensorizer at `k lanes 32`: the recipe enumerates that window
    /// (tile 6) and nothing else exercises it. It is the only tile that halves
    /// the shared buffer to 16 KB, which is what two blocks per SM needs — but
    /// the recipe's config for it produces wrong values, so this pins whether
    /// the four-step window is correct in the tensorizer at all, without the
    /// recipe in the way.
    #[test]
    fn staged_four_step_window_values() {
        const M: i64 = 128;
        const N: i64 = 128;
        // `k lanes 32` over a K loop that is a whole number of `k8` steps per
        // lane: 32 lanes × 8 halves = 256, i.e. 8 chunks of 32 halves.
        const K: i64 = 256;

        let build = || {
            let mut k = Kernel::new(Dev::Auto);
            let mut info = cuda();
            info.local_mem_size = 49152;
            k.dev_info = Some(Arc::new(info));
            let a = k.param(DType::F16);
            let b = k.param(DType::F16);
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
            let c_idx = k.mad(m, N, n);
            k.store(c32, res, c_idx);
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
            let out = compiled.forward(&[a, b], vec![[M, N]]).ok()?;
            Some(out[0].clone().try_into().unwrap())
        };

        let (mut ks, ms, ns, kls) = build();
        assert!(!ks.split(ms, split::SplitMode::Unroll, 128).is_null(), "staged m split");
        assert!(!ks.split(ns, split::SplitMode::Unroll, 128).is_null(), "staged n split");
        assert!(!ks.split(kls, split::SplitMode::Unroll, 32).is_null(), "staged k split");
        ks.verify();
        let sopt = ks.opt_tensorize();
        assert!(sopt.nconfigs() > 0, "the four-step window is a staged config");
        sopt.apply(&mut ks, plan_config(0, 0, true));
        ks.default_epilogue();
        ks.verify();

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
        let Some(got) = run(ks, &a, &b) else { return };
        let mut worst = 0.0f32;
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut sum = 0.0f32;
                for t in 0..K as usize {
                    sum += a_host[i * K as usize + t] * b_host[j * K as usize + t];
                }
                worst = worst.max((got[i * N as usize + j] - sum).abs());
            }
        }
        println!("four-step staged vs host: max |err| = {worst}");
        assert!(worst < 1e-2, "the four-step staged window disagrees with the host: {worst}");
    }

    /// llama.cpp's `mul_mat_f` form: a warp-private tile of 16 rows refilled
    /// per operand sub-tile, 32 output rows per block, K split across the
    /// warps, and the shared combine epilogue. Values must match a host dot
    /// product.
    #[test]
    fn llama_form_values() {
        const M: i64 = 16;
        const N: i64 = 32;
        const K: i64 = 128;

        let mut k = Kernel::new(Dev::Auto);
        let mut info = cuda();
        info.local_mem_size = 49152;
        k.dev_info = Some(Arc::new(info));
        let a = k.param(DType::F16);
        let b = k.param(DType::F16);
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
        let c_idx = k.mad(m, N, n);
        k.store(c32, res, c_idx);

        // 16 output columns, 32 output rows (one warp), one warp_size of
        // half2 per K step: llama.cpp's `mul_mat_f` instantiation.
        assert!(!k.split(m, split::SplitMode::Unroll, 16).is_null(), "m split");
        assert!(!k.split(n, split::SplitMode::Unroll, 32).is_null(), "n split");
        assert!(!k.split(k_loop, split::SplitMode::Unroll, 64).is_null(), "k split");
        k.verify();
        let opt = k.opt_tensorize();
        // The split kernel carries one pre-tiled shape, so shape index 0;
        // layout 10 is `(1, 1, 2)`, the K split that sums per-warp partials
        // through the combine buffer.
        let combine = plan_config(0, 10, true);
        opt.apply(&mut k, combine);
        k.default_epilogue();
        k.devectorize().expect("llama form devectorizes");
        k.verify();
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::PTX(p) if matches!(p.as_ref(), PTXOp::WarpSync))),
            "the warp-private tile syncs its own warp"
        );

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
        let compiled = match k.compile() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let out = compiled.forward(&[&a, &b], vec![[M, N]]).unwrap();
        let got: Vec<f32> = out[0].clone().try_into().unwrap();
        let mut worst = 0.0f32;
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut sum = 0.0f32;
                for t in 0..K as usize {
                    sum += a_host[i * K as usize + t] * b_host[j * K as usize + t];
                }
                worst = worst.max((got[i * N as usize + j] - sum).abs());
            }
        }
        println!("llama form vs host: max |err| = {worst}");
        assert!(worst < 1e-2, "llama-form tensorize disagrees with the host dot product: {worst}");
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

        let recipe = k.opt_tensorize();
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
        // One matched region, thirteen layouts, two operand forms, with the
        // split-K layouts offered staged only.
        assert_eq!(t.nconfigs(), 115, "one region and four tile grids, per layout and operand form");
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

        let wmma = k.iter_unordered().find_map(|(id, op)| matches!(op, Op::Wmma { .. }).then_some(id)).expect("wmma present");
        let Op::Wmma { a: a_frag, b: b_frag, .. } = *k.at(wmma) else {
            unreachable!()
        };
        for operand in [a_frag, b_frag] {
            let Op::Stack { ref ops } = *k.at(operand) else {
                panic!("operand is not a stack")
            };
            for &slot in ops.iter() {
                let Op::Load { src, .. } = *k.at(slot) else {
                    panic!("slot is not a load")
                };
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
        assert_eq!(opt.nconfigs(), 1403, "one region and sixty tile grids, per layout and operand form");
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
        k.devectorize().unwrap();

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
