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
    backend::{DeviceInfo, FragAxis, FragMap, MMAVariant},
    dtype::Constant,
    kernel::{
        BOp, Kernel, LdMatrixDims, LdMatrixNum, LdMatrixTrans, LdMatrixType, MMADType, MemScope, Op, OpId, PTXOp, ParamKind,
        RangeKind, autotune::Optimization, split::SplitMode,
    },
    shape::Dim,
};

/// `ldmatrix`'s own geometry: an `m8n8` matrix is an eight-by-eight element
/// block, and the staged tile packs k-blocks of exactly that many elements
/// (`LDMATRIX_BLOCK`). The address arithmetic below shifts and masks by
/// `LDMATRIX_LOG2` / `LDMATRIX_EDGE - 1` because the instruction's shape says
/// so, never because the matmul shape does.
const LDMATRIX_EDGE: Dim = 8;
const LDMATRIX_LOG2: i64 = 3;
const LDMATRIX_BLOCK: Dim = LDMATRIX_EDGE * LDMATRIX_EDGE;

/// The value of one product operand, as the tensorization sees it.
///
/// A plain operand is a single `Load` of its storage. A *quantized* operand is
/// a chain of ops computed from its packed storage — a dequant — and the
/// product multiplies that chain's value, not a load. [`Cone`] records the
/// chain in a form the fill can rebuild element by element, without knowing
/// which quantization format it is: everything the chain computes is a
/// function of the two coordinates the operand's element index is composed
/// of — the output row/column and the contraction element — so substituting
/// those two leaves with an element's own coordinates re-derives the whole
/// chain for that element (a block index, a code's byte, a super-block's
/// scale row, a per-group scale's address, all of it).
///
/// This is what lets one matcher admit both a bare `mul(load, load)` and a
/// fused dequant: the operand's *logical* view (its `m`/`k` or `n`/`k`
/// strides, its dtype) is read off `root` and the chain's value exactly as it
/// is read off the load's index, and only the fill differs.
#[derive(Debug, Clone)]
struct Cone {
    /// The chain's value: the op the product multiplies.
    value: OpId,
    /// The affine element index the chain is rooted at (the view analysis reads
    /// this, the fill does not: the leaves below are what it substitutes).
    root: OpId,
    /// Every computed op of the chain in dependency order (leaves excluded, so
    /// params, storages, constants, ranges and lanes are referenced, not
    /// cloned).
    ops: Vec<(OpId, Op)>,
    /// The row/column leaf and the contraction leaf the chain reads, and the
    /// contraction's extent: [`Cone::clone_at`] replaces them with the
    /// element's coordinates.
    mn: OpId,
    k_leaf: OpId,
    k_extent: Dim,
}

impl Cone {
    /// Every op the chain can be computed from, in dependency order, leaves
    /// (params, storages, constants, ranges, lanes) excluded: those are
    /// referenced, never cloned.
    fn reach(k: &Kernel, value: OpId) -> Vec<OpId> {
        let mut out = Vec::new();
        let mut seen: Set<OpId> = Set::default();
        Cone::walk(k, value, &mut seen, &mut out);
        out
    }

    fn walk(k: &Kernel, id: OpId, seen: &mut Set<OpId>, out: &mut Vec<OpId>) {
        if id.is_null() || !seen.insert(id) {
            return;
        }
        let op = k.at(id);
        if !Cone::is_leaf(op) {
            for p in op.parameters() {
                Cone::walk(k, p, seen, out);
            }
            out.push(id);
        }
    }

    /// Whether an op is a leaf of the cone (a definition that is reused, not
    /// part of the computed chain).
    fn is_leaf(op: &Op) -> bool {
        matches!(
            op,
            Op::Param { .. }
                | Op::Storage { .. }
                | Op::Const(_)
                | Op::Range { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Unroll { .. }
        )
    }

    /// The cone behind `value`, or `None` when it is a single load (the plain
    /// form) or when its chains do not root at exactly one affine index over
    /// the output and contraction ranges.
    fn at(k: &Kernel, value: OpId) -> Option<Cone> {
        let order = Cone::reach(k, value);
        // The root: the affine index expression every other affine part of the
        // chain is derived from (a block index divides it, a byte masks it, a
        // scale row strides it). Among the chain's affine ops it is the one
        // with the *fewest* producers, and all the others must be computed
        // from it: the chain is rooted at exactly one index expression, whether
        // that names two axes (a scalar loop) or more (lanes the splits added).
        let cands: Vec<OpId> = order
            .iter()
            .copied()
            .filter(|&id| k.get_strides(id).is_some_and(|m| m.keys().filter(|v| !v.is_null()).count() >= 2))
            .collect();
        if cands.is_empty() {
            if std::env::var("ZYX_TD").is_ok() {
                eprintln!(
                    "TD cone: value={:?} no affine candidate among {} ops: {:?}",
                    k.at(value),
                    order.len(),
                    order.iter().map(|&id| format!("{:?}", k.at(id))).collect::<Vec<_>>()
                );
            }
            return None;
        }
        let vars = |id: OpId| -> usize { k.get_strides(id).map(|m| m.keys().filter(|v| !v.is_null()).count()).unwrap_or(0) };
        let maxv = cands.iter().map(|&c| vars(c)).max()?;
        let top: Vec<OpId> = cands.iter().copied().filter(|&c| vars(c) == maxv).collect();
        if top.len() != 1 {
            if std::env::var("ZYX_TD").is_ok() {
                eprintln!("TD cone: {} top candidates for value {:?}", top.len(), k.at(value));
            }
            return None;
        }
        let root = top[0];
        if root == value {
            return None;
        }
        let (mn, k_leaf, k_extent) = element_leaves(k, root)?;
        // Everything the chain computes must be a function of the element's two
        // coordinates: any other varying leaf (a lane, a loop or a range) would
        // be read at the tile's first element for every element.
        let dead: Set<OpId> = Cone::reach(k, mn).into_iter().chain(Cone::reach(k, k_leaf)).collect();
        for &id in &order {
            if dead.contains(&id) {
                continue;
            }
            for p in k.at(id).parameters() {
                if matches!(k.at(p), Op::Range { .. } | Op::Loop { .. } | Op::Unroll { .. }) && p != mn && p != k_leaf {
                    return None;
                }
            }
        }
        let ops: Vec<(OpId, Op)> = order.into_iter().map(|id| (id, k.at(id).clone())).collect();
        if ops.last().map(|(id, _)| *id) != Some(value) {
            return None;
        }
        Some(Cone { value, root, ops, mn, k_leaf, k_extent })
    }

    /// The element's two coordinates from its linear index: the row/column and
    /// the contraction element.
    fn coords(&self, kernel: &mut Kernel, before: OpId, index: OpId) -> (OpId, OpId) {
        let put = |kernel: &mut Kernel, op: Op| -> OpId {
            if before.is_null() {
                kernel.push_back(op)
            } else {
                kernel.insert_before(before, op)
            }
        };
        let ke = put(kernel, Op::Const(Constant::idx(self.k_extent)));
        let row = put(kernel, Op::Binary { x: index, y: ke, bop: BOp::Div });
        let ke = put(kernel, Op::Const(Constant::idx(self.k_extent)));
        let kk = put(kernel, Op::Binary { x: index, y: ke, bop: BOp::Mod });
        (row, kk)
    }

    /// Rebuild the cone in `kernel` at the element whose linear index is
    /// `index`: every op re-derived from that element's coordinates. Returns
    /// the value.
    fn clone_at(&self, kernel: &mut Kernel, before: OpId, index: OpId) -> OpId {
        let (row, kk) = self.coords(kernel, before, index);
        self.clone_at_coords(kernel, before, row, kk)
    }

    /// Rebuild the cone at the element whose row/column coordinate and
    /// contraction coordinate are already known. A fill that computed the two
    /// from the tile's own geometry hands them over instead of a linear index,
    /// which the coordinate split would have to divide back out of it.
    fn clone_at_coords(&self, kernel: &mut Kernel, before: OpId, row: OpId, kk: OpId) -> OpId {
        self.clone_op_at_coords(kernel, before, row, kk, self.value)
    }

    /// [`Cone::clone_at_coords`] returning the clone of one op of the chain
    /// instead of its value: a fill that stages a *word* of the operand needs
    /// the chain's own load, re-derived at the element's coordinates.
    fn clone_op_at_coords(&self, kernel: &mut Kernel, before: OpId, row: OpId, kk: OpId, target: OpId) -> OpId {
        if std::env::var("ZYX_CONE").is_ok() {
            eprintln!(
                "CONE mn={:?} k={:?} extent={:?} value={:?} ops={:?}",
                self.mn,
                self.k_leaf,
                self.k_extent,
                self.value,
                self.ops.iter().map(|(id, op)| format!("{id:?}={op:?}")).collect::<Vec<_>>()
            );
        }
        let mut remap: Map<OpId, OpId> = Map::default();
        remap.insert(self.mn, row);
        remap.insert(self.k_leaf, kk);
        // What the clone still needs: the value's cone with the two substituted
        // leaves standing for themselves. Everything reachable only through a
        // substituted leaf is the *coordinate's* own cone — the caller hands
        // that in already (`row`, `kk`), so cloning it would re-derive the
        // composition the substitution replaces, and a member of it can read a
        // loop the clone is placed outside of (a split's `loop * step + lane`
        // contraction index is one: a fill anchored before the loop would then
        // read the loop it precedes).
        let mut need: Set<OpId> = Set::default();
        {
            let mut stack = vec![self.value];
            while let Some(id) = stack.pop() {
                if id.is_null() || id == self.mn || id == self.k_leaf || !need.insert(id) {
                    continue;
                }
                stack.extend(kernel.at(id).parameters());
            }
        }
        for (id, op) in &self.ops {
            // The two substituted leaves are not cloned: their coordinates
            // already stand for them, and cloning them would re-derive the
            // composition from the very lanes the substitution replaces.
            if *id == self.mn || *id == self.k_leaf || !need.contains(id) {
                continue;
            }
            let mut op = op.clone();
            for p in op.parameters_mut() {
                if let Some(&new) = remap.get(p) {
                    *p = new;
                }
            }
            let new = if before.is_null() {
                kernel.push_back(op)
            } else {
                kernel.insert_before(before, op)
            };
            remap.insert(*id, new);
        }
        remap[&target]
    }

    /// The length of the aligned index run over which this value is constant:
    /// `1` for anything that moves with the index, the divisor's own length
    /// for a `Div` of the index, the smaller of the two for a sum or product,
    /// and unbounded for a value the index does not reach at all.
    ///
    /// This is what says a quantized operand's scale has one value per group:
    /// the group size divides the run, so the scale is the same for every
    /// element of the group and the int8 form can hoist it out of the
    /// contraction.
    /// `mn` and `k` are the two leaves the operand's element index is composed
    /// of, and `k_extent` the contraction's extent: the row/column leaf changes
    /// once per `k_extent` elements, the contraction leaf once per element.
    fn period(kern: &Kernel, id: OpId, mn: OpId, k: OpId, k_extent: Dim, depth: usize) -> Option<Dim> {
        if depth > 64 {
            return None;
        }
        if id == mn {
            return Some(k_extent);
        }
        if id == k {
            return Some(Dim::from(1));
        }
        let op = kern.at(id);
        if let Op::Binary { x, y, bop: BOp::Div } = op
            && let Op::Const(c) = kern.at(*y)
            && c.as_dim().is_some_and(|d| d > 0)
        {
            // `index / c` only matters for the index itself: if the numerator
            // moves with the index, the quotient is constant on `c`-aligned
            // runs.
            let p = Cone::period(kern, *x, mn, k, k_extent, depth + 1)?;
            let c = c.as_dim()?;
            return Some(if p == Dim::from(1) { c } else { p.min(c) });
        }
        match op {
            Op::Param { .. } | Op::Storage { .. } | Op::Const(_) => Some(Dim::from(i64::MAX)),
            Op::Range { .. } | Op::Loop { .. } | Op::Unroll { .. } => Some(Dim::from(1)),
            Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Unary { x, .. } => {
                Cone::period(kern, *x, mn, k, k_extent, depth + 1)
            }
            Op::Index { vec, .. } | Op::Copy { src: vec, .. } => Cone::period(kern, *vec, mn, k, k_extent, depth + 1),
            Op::Load { src_idx, .. } => Cone::period(kern, *src_idx, mn, k, k_extent, depth + 1),
            Op::Select { if_true, .. } => Cone::period(kern, *if_true, mn, k, k_extent, depth + 1),
            Op::Binary { x, y, bop } => match bop {
                // The index itself: nothing divides it.
                BOp::Mod => Some(Dim::from(1)),
                BOp::Mul => match (kern.at(*x), kern.at(*y)) {
                    (Op::Const(_), _) => Cone::period(kern, *y, mn, k, k_extent, depth + 1),
                    (_, Op::Const(_)) => Cone::period(kern, *x, mn, k, k_extent, depth + 1),
                    _ => {
                        let a = Cone::period(kern, *x, mn, k, k_extent, depth + 1)?;
                        let b = Cone::period(kern, *y, mn, k, k_extent, depth + 1)?;
                        Some(a.min(b))
                    }
                },
                _ => {
                    let a = Cone::period(kern, *x, mn, k, k_extent, depth + 1)?;
                    let b = Cone::period(kern, *y, mn, k, k_extent, depth + 1)?;
                    Some(a.min(b))
                }
            },
            _ => Some(Dim::from(1)),
        }
    }
}

/// The int8 form's view of a cone: the operand split into the integer code the
/// `m8n8k16` instruction multiplies and the per-group scale the epilogue
/// applies to its int32 accumulator (llama.cpp's `q8_1`/`pq2_0` structure).
///
/// The split is structural, not format-specific: the cone's value is a product
/// of two factors, and one of them is constant over the int8 instruction's
/// group ([`Q8_1_GROUP`]) — that factor is the scale, the other is the code.
/// The code is rebuilt at each element's own index; the scale is rebuilt at a
/// group's representative element, from the two leaves the element index is
/// composed of (the row/column leaf and the contraction leaf).
#[derive(Debug, Clone)]
struct Quant {
    /// The code's cone: its value cast to int8 is the operand word's byte.
    code: Cone,
    /// The scale's cone: the value cone restricted to what the scale reads, so
    /// it is rebuilt at a group's representative element.
    scale: Cone,
}

/// Elements per int8 group: llama.cpp's `q8_1` block, and the run the weight
/// block scales are replicated over. Two `m8n8k16` K steps.
const Q8_1_GROUP: Dim = 32;

impl Quant {
    /// The int8 split of `cone`, or `None` when its value is not the product
    /// of an integer code and a factor that is constant over the int8 group.
    fn at(k: &Kernel, cone: &Cone) -> Option<Quant> {
        // The cone's value may be wrapped in the casts the product applies
        // (the linearized dequant casts to f16, the f16 product casts to f32).
        let mut v = cone.value;
        while let Op::Cast { x, .. } | Op::Bitcast { x, .. } = k.at(v) {
            v = *x;
        }
        let Op::Binary { x, y, bop: BOp::Mul } = *k.at(v) else {
            return None;
        };
        // Either multiplicand may be the scale.
        for (s, c) in [(x, y), (y, x)] {
            let Some(p) = Cone::period(k, s, cone.mn, cone.k_leaf, cone.k_extent, 0) else {
                continue;
            };
            if p < Q8_1_GROUP {
                continue;
            }
            let Some(code) = Cone::at(k, c) else {
                continue;
            };
            if code.root != cone.root {
                continue;
            }
            // The scale's own ops inside the value's cone: everything it is
            // computed from that the cone clones (the leaves it reads stay
            // leaves, and are substituted by coordinates like the value's).
            let below: Set<OpId> = Cone::reach(k, s).into_iter().collect();
            let scale_ops: Vec<(OpId, Op)> = cone.ops.iter().filter(|(id, _)| below.contains(id)).cloned().collect();
            if !scale_ops.iter().any(|(id, _)| *id == s) {
                continue;
            }
            let scale =
                Cone { value: s, root: cone.root, ops: scale_ops, mn: cone.mn, k_leaf: cone.k_leaf, k_extent: cone.k_extent };
            return Some(Quant { code, scale });
        }
        None
    }
}

/// An operand's code as one packed word, the form the int8 fill stages: the
/// code's cone is `((word >> bits*(elem % codes)) & (2^bits - 1)) - zero` over
/// a single loaded element. `codes` is how many codes that element holds, so
/// the fill loads it once and writes `codes / 4` operand words.
///
/// This is the *shape* of a packed quantization, not a format: a two-bit code
/// in a sixteen-bit word and a byte-addressable int8 code in a thirty-two-bit
/// word are the same description at different widths.
#[derive(Clone, Copy)]
struct CodeWord {
    /// The cone's own word load, rebuilt at an element's coordinates by
    /// [`Cone::clone_op_at_coords`].
    load: OpId,
    /// Codes per loaded element.
    codes: u32,
    /// Bits per code.
    bits: u32,
    /// The constant the chain subtracts (`code - zero`).
    zero: i32,
    /// The load's element dtype.
    dtype: crate::DType,
}

impl Quant {
    /// The code's packed-word form, or `None` when the code is not a fixed-width
    /// field of a single loaded element.
    fn code_word(&self, k: &Kernel) -> Option<CodeWord> {
        let strip = |mut v: OpId| -> OpId {
            loop {
                match *k.at(v) {
                    Op::Cast { x, .. } | Op::Bitcast { x, .. } => v = x,
                    _ => return v,
                }
            }
        };
        let con = |id: OpId| -> Option<i64> {
            match k.at(strip(id)) {
                Op::Const(c) => c.as_integer(),
                _ => None,
            }
        };
        let mut v = strip(self.code.value);
        // The centering the chain applies to the code (`code - zero`).
        let zero = if let Op::Binary { x, y, bop: BOp::Sub } = *k.at(v) {
            let Op::Const(c) = k.at(y) else {
                return None;
            };
            let z = c.as_float()?;
            if z.fract() != 0.0 {
                return None;
            }
            v = strip(x);
            z as i32
        } else {
            0
        };
        // The code's field: `(word >> shift) & mask`.
        let Op::Binary { x, y, bop: BOp::BitAnd } = *k.at(v) else {
            return None;
        };
        let Op::Const(mc) = k.at(y) else {
            return None;
        };
        let mask = mc.as_integer()?;
        let Some(span) = u64::try_from(mask + 1).ok().filter(|m| m.is_power_of_two()) else {
            return None;
        };
        let bits = u32::try_from(span.trailing_zeros()).ok()?;
        let Op::Binary { x: sh, y: shf, bop: BOp::BitShiftRight } = *k.at(strip(x)) else {
            return None;
        };
        // The shift is `bits * (elem % codes)`: a constant stride times the
        // element's position inside the word.
        let Op::Binary { x: sx, y: sy, bop: BOp::Mul } = *k.at(strip(shf)) else {
            return None;
        };
        let (step, mob) = match (con(sx), con(sy)) {
            (Some(c), None) => (c, sy),
            (None, Some(c)) => (c, sx),
            _ => return None,
        };
        if step != i64::from(bits) {
            return None;
        }
        let Op::Binary { x: mx, y: my, bop: BOp::Mod } = *k.at(strip(mob)) else {
            return None;
        };
        let codes = u32::try_from(con(my).or_else(|| con(mx))?).ok()?;
        if codes == 0 || codes % 4 != 0 {
            return None;
        }
        let word = strip(sh);
        let Op::Load { .. } = *k.at(word) else {
            return None;
        };
        Some(CodeWord { load: word, codes, bits, zero, dtype: k.dtype(word) })
    }

    /// Whether the loaded element's bytes are the codes themselves, so the fill
    /// copies the word instead of unpacking it.
    fn byte_word(&self, w: &CodeWord) -> bool {
        w.bits == 8 && w.codes == 4 && w.dtype == DType::U32 && w.zero == 0
    }
}

/// The two leaves an operand's element index is composed of: the row/column
/// leaf (of the output range) and the contraction leaf, with the contraction's
/// extent. The linearizer writes the view as `mn * K + k` (`mad`), in either
/// order.
fn element_leaves(k: &Kernel, root: OpId) -> Option<(OpId, OpId, Dim)> {
    let Op::Binary { x, y, bop: BOp::Add } = *k.at(root) else {
        return None;
    };
    let strip = |id: OpId| -> (OpId, Option<Dim>) {
        if let Op::Binary { x, y, bop: BOp::Mul } = *k.at(id) {
            if let Op::Const(c) = k.at(y)
                && let Some(d) = c.as_dim()
            {
                return (x, Some(d));
            }
            if let Op::Const(c) = k.at(x)
                && let Some(d) = c.as_dim()
            {
                return (y, Some(d));
            }
        }
        (id, None)
    };
    let (xa, ka) = strip(x);
    let (xb, kb) = strip(y);
    match (ka, kb) {
        (Some(k_extent), None) => Some((xa, y, k_extent)),
        (None, Some(k_extent)) => Some((x, xb, k_extent)),
        _ => None,
    }
}

/// The int8 split of an operand value, when its chain exposes one.
fn cone_quant(k: &Kernel, value: OpId) -> Option<Quant> {
    operand_view(k, value, true).and_then(|(_, cone, _)| cone).and_then(|c| Quant::at(k, &c))
}
/// The operand behind one side of the product: the storage a plain load reads
/// and that load's index, or — for a fused dequant — the chain's cone, with
/// the affine index op the chain is rooted at. `global_only` also requires
/// every storage the operand reads to be a global parameter (the tiled form
/// only ever reads global operands).
fn operand_view(k: &Kernel, value: OpId, global_only: bool) -> Option<(OpId, Option<Cone>, OpId)> {
    let global = |src: OpId| matches!(k.at(src), Op::Param { kind: ParamKind::Global, .. });
    if let Op::Load { src, src_idx } = *k.at(value) {
        if std::env::var("ZYX_TD").is_ok() && global_only && !global(src) {
            eprintln!("TD operand_view: load src {:?} is not global", k.at(src));
        }
        if global_only && !global(src) {
            return None;
        }
        return Some((src, None, src_idx));
    }
    let cone = Cone::at(k, value)?;
    if global_only {
        for (_, op) in &cone.ops {
            if let Op::Load { src, .. } = op
                && !global(*src)
            {
                return None;
            }
        }
    }
    Some((OpId::NULL, Some(cone.clone()), cone.root))
}

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
    /// The operand's value when it is a fused dequant chain rather than a
    /// load, and that chain: `a_src` is then a null storage and the fill
    /// rebuilds the cone instead of loading the view's element.
    a_cone: Option<Cone>,
    a_m: Dim,
    a_k: Dim,
    b_src: OpId,
    b_cone: Option<Cone>,
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
    /// Fused dequant chains in place of a plain load, and their int8 split
    /// when they expose one (see [`Cone`] / [`Quant`]). The fills rebuild the
    /// chain per tile element instead of loading the view's element.
    a_cone: Option<Cone>,
    b_cone: Option<Cone>,
    a_quant: Option<Quant>,
    b_quant: Option<Quant>,
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
    /// The operand roles are the transpose of the matched product order:
    /// `a_src` is the second operand and `b_src` the first, so `mt` counts
    /// tiles along the second operand's rows.
    mirrored: bool,
}

/// Warp layouts a tensorized block may use, as `(m, n, k)` warps over the
/// output rows, the output columns, and the contraction. One axis at a time:
/// a warp grid owns disjoint output sub-tiles (nothing is summed), a K split
/// sums the per-warp partials in shared memory.
pub(crate) const WARP_LAYOUTS: [(u16, u16, u16); 18] = [
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
    // Sixteen-warp blocks: 512 threads, the shape the int8 mainloop wants
    // (a 128x64 block is eight 16-row bands by two 32-column bands).
    (16, 1, 1),
    (8, 2, 1),
    (4, 4, 1),
    (2, 8, 1),
    (1, 16, 1),
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
    /// the lane factors to split them by, the instruction tile those factors
    /// are multiples of, and whether the plan wants the transposed tile
    /// extents (its m lanes are the emitted form's n).
    grid: Option<(OpId, OpId, OpId, u16, u16, u16, Dim, Dim, Dim, bool)>,
    layout: (u16, u16, u16),
    /// Stage the operands through shared memory (`ldmatrix`) instead of
    /// reading their fragments from global memory.
    staged: bool,
    /// The staged layout: the swizzled one (bank-conflict-free rows) or the
    /// plain one. Only the s8 form has both today; every other form has one.
    swizzle: bool,
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
        if std::env::var("ZYX_PLANS").is_ok() {
            for (i, p) in self.plans.iter().enumerate() {
                let form =
                    p.region.map(|r| self.regions[r].variant.dtype).or_else(|| p.tiled.map(|t| self.tiled[t].variant.dtype));
                eprintln!(
                    "PLAN {i}: region={:?} tiled={:?} grid={:?} layout={:?} staged={} swizzle={} dtype={form:?}",
                    p.region, p.tiled, p.grid, p.layout, p.staged, p.swizzle
                );
            }
        }
        let Some(plan) = self.plans.get(config as usize) else {
            return;
        };
        if let Some((m, n, k, m_lanes, n_lanes, k_lanes, fm, fn_, fk, transposed)) = plan.grid {
            // The grid is this plan's own state: the predicates proved every
            // factor divides and every split applies, so the splits apply. No
            // clone, no probe. A split that still refuses (the render ceiling
            // counts the IR the earlier splits grew) returns NULL without
            // touching the kernel; the candidate is then the valid
            // partially-split kernel, not a tiling this plan promised.
            let sm = kernel.split(m, SplitMode::Unroll, m_lanes);
            let sn = if sm.is_null() {
                OpId::NULL
            } else {
                kernel.split(n, SplitMode::Unroll, n_lanes)
            };
            let sk = if sn.is_null() {
                OpId::NULL
            } else {
                kernel.split(k, SplitMode::Unroll, k_lanes)
            };
            if std::env::var("ZYX_KP").is_ok() {
                eprintln!(
                    "SPLIT cfg={config} ops={} m={m_lanes} sm={} sn={} sk={} ops_after={}",
                    kernel.ops.len(),
                    !sm.is_null(),
                    !sn.is_null(),
                    !sk.is_null(),
                    kernel.ops.len()
                );
            }
            if sm.is_null() || sn.is_null() || sk.is_null() {
                return;
            }
            // The tile grid this plan asked for, in the orientation it named:
            // a transposed plan's m lanes are the form's n lanes.
            let want = if transposed {
                (Dim::from(n_lanes) / fm, Dim::from(m_lanes) / fn_, Dim::from(k_lanes) / fk)
            } else {
                (Dim::from(m_lanes) / fm, Dim::from(n_lanes) / fn_, Dim::from(k_lanes) / fk)
            };
            let (_, forms) = kernel.tensor_forms();
            if std::env::var("ZYX_MT").is_ok() {
                eprintln!(
                    "ZYX_MT want={want:?} forms={:?}",
                    forms
                        .iter()
                        .map(|t| (t.mt, t.nt, t.kt, t.variant.dtype, t.a_cone.is_some(), t.b_cone.is_some()))
                        .collect::<Vec<_>>()
                );
            }
            // The form those extents name. The matcher labels the extents by
            // the *product's* operand order, so for a seed whose product is
            // written `b * a` the plan's m axis is the form's n: the extents
            // alone find it, whichever operand roles the form was built with.
            let matches: Vec<&Tiled> = forms.iter().filter(|t| (t.mt, t.nt, t.kt) == want).collect();
            let chosen = match matches.len() {
                0 => panic!(
                    "the enumerated tile grid must be derivable from its own splits: cfg={config} lanes {m_lanes}x{n_lanes}x{k_lanes} tile {fm:?}x{fn_:?}x{fk:?} transposed={transposed} want={want:?}"
                ),
                // Equal lane lengths give both operand forms the same extents:
                // the plan's orientation picks which one this config is.
                1 => matches[0],
                _ => *matches.iter().find(|t| t.mirrored == transposed).unwrap_or(&matches[0]),
            };
            tile_to_wmma_form(kernel, chosen, plan.staged, plan.swizzle, plan.layout);
            if std::env::var("ZYX_KP").is_ok() {
                eprintln!(
                    "ZYX_KP cfg={config} m_lanes={} n_lanes={} k_lanes={} layout={:?} staged={} ops={}",
                    m_lanes,
                    n_lanes,
                    k_lanes,
                    plan.layout,
                    plan.staged,
                    kernel.ops.len()
                );
            }
            kernel.verify();
            return;
        }
        let staged = plan.staged;
        let layout = plan.layout;
        if let Some(i) = plan.tiled {
            tile_to_wmma_form(kernel, &self.tiled[i], staged, plan.swizzle, layout);
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
                let (asl, av) = match &r.a_cone {
                    Some(cone) => coop_fill_cone(k, OpId::NULL, lane, cone, a_base, r.a_m, r.a_k, fm, fk, threads, plan.swizzle),
                    None => coop_fill_load(k, OpId::NULL, lane, r.a_src, a_base, r.a_m, r.a_k, fm, fk, threads, plan.swizzle),
                };
                coop_fill_store(k, OpId::NULL, a_tile, OpId::NULL, asl, av);
                let (bsl, bv) = match &r.b_cone {
                    Some(cone) => coop_fill_cone(k, OpId::NULL, lane, cone, b_base, r.b_n, r.b_k, fn_, fk, threads, plan.swizzle),
                    None => coop_fill_load(k, OpId::NULL, lane, r.b_src, b_base, r.b_n, r.b_k, fn_, fk, threads, plan.swizzle),
                };
                coop_fill_store(k, OpId::NULL, b_tile, OpId::NULL, bsl, bv);
                k.push_back(Op::Barrier);
                let a = coop_ldmatrix(
                    k,
                    OpId::NULL,
                    lane,
                    a_tile,
                    fm,
                    fk,
                    fm,
                    0,
                    fk / LDMATRIX_EDGE,
                    OpId::NULL,
                    OpId::NULL,
                    plan.swizzle,
                );
                let b = coop_ldmatrix(
                    k,
                    OpId::NULL,
                    lane,
                    b_tile,
                    fn_,
                    fk,
                    fn_,
                    0,
                    fk / LDMATRIX_EDGE,
                    OpId::NULL,
                    OpId::NULL,
                    plan.swizzle,
                );
                (a, b)
            } else {
                let mut a_elems = Vec::with_capacity(v.a.slots.len());
                for &slot in &v.a.slots {
                    let idx = frag_index(k, &v.a, (m_base, k_base), (r.a_m, r.a_k), slot);
                    a_elems.push(match &r.a_cone {
                        Some(cone) => cone.clone_at(k, OpId::NULL, idx),
                        None => k.load(r.a_src, idx),
                    });
                }
                let a_frag = k.stack(&a_elems);
                let mut b_elems = Vec::with_capacity(v.b.slots.len());
                for &slot in &v.b.slots {
                    let idx = frag_index(k, &v.b, (n_base, k_base), (r.b_n, r.b_k), slot);
                    b_elems.push(match &r.b_cone {
                        Some(cone) => cone.clone_at(k, OpId::NULL, idx),
                        None => k.load(r.b_src, idx),
                    });
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
    cone: Option<&Cone>,
    base: OpId,
    sr: Dim,
    sc: Dim,
    warp: u32,
) -> Vec<OpId> {
    // One direct fragment load carries 8 bytes — the `ldg8` the reference's
    // unstage kernel issues — so the element count follows the operand's own
    // width and is capped by the widest width the device renders. A cone has
    // no contiguous view to load: its value's dtype owns the element width,
    // and the run only groups its per-slot chains.
    const ACCESS_BYTES: u32 = 8;
    let dtype = match cone {
        Some(cone) => kernel.dtype(cone.value),
        None => {
            debug_assert!(!src.is_null(), "frag_elems: neither a source nor a cone");
            kernel.dtype(src)
        }
    };
    let per_access = ACCESS_BYTES * 8 / u32::from(dtype.bit_size());
    let max_vec = usize::from(
        kernel.device_info().supported_vec_lens.iter().copied().filter(|&v| u32::from(v) <= per_access).max().unwrap_or(1),
    );
    let slots = &map.slots;
    let mut out = Vec::with_capacity(slots.len());
    let mut i = 0;
    while i < slots.len() {
        // The longest run of slots that is contiguous in memory, up to one
        // vector access. Its innermost lane is the vector width:
        // `devectorize` keeps it and the backend renders one vector load,
        // which the `Stack` then reads lane by lane through `Index`.
        let mut n = 1;
        while i + n < slots.len() && n < max_vec {
            let a = slots[i + n - 1];
            let b = slots[i + n];
            let delta = Dim::from(b.0 - a.0) * sr + Dim::from(b.1 - a.1) * sc;
            if delta != 1 {
                break;
            }
            n += 1;
        }
        let addr = slot_addr(kernel, before, lane, map, base, sr, sc, slots[i], warp);
        if let Some(cone) = cone {
            // A fused dequant has no vector form to load: the slot's own
            // element, one chain per slot.
            for s in 0..n {
                let at = if s == 0 {
                    addr
                } else {
                    let off = kernel.insert_before(before, Op::Const(Constant::idx(Dim::from(s as i64))));
                    kernel.insert_before(before, Op::Binary { x: addr, y: off, bop: BOp::Add })
                };
                out.push(cone.clone_at(kernel, before, at));
            }
        } else {
            // One scalar load per slot, never one vector load sliced by
            // `Index`: the epilogue expands the vector load's lanes into
            // scalars while the fragment's `Index` ops still address it, and an
            // `Index` of a scalar is not renderable. The slots are the same
            // addresses the vector would have covered.
            for s in 0..n {
                let at = if s == 0 {
                    addr
                } else {
                    let off = kernel.insert_before(before, Op::Const(Constant::idx(Dim::from(s as i64))));
                    kernel.insert_before(before, Op::Binary { x: addr, y: off, bop: BOp::Add })
                };
                out.push(kernel.insert_before(before, Op::Load { src, src_idx: at }));
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
    swizzle: bool,
) -> (OpId, OpId) {
    coop_fill_load_with(kernel, before, lane, src, None, base, sr, sc, rows, chunk, threads, swizzle)
}

/// [`coop_fill_load`] for an operand that is a fused dequant chain rather than
/// a plain load: the same slot layout, the same step lane, the same vector
/// width — but the value is the chain rebuilt per element (the vector lane is
/// the element index), since a packed operand has no contiguous view to load.
/// `devectorize` unfolds the vector lane into one element per instruction.
#[allow(clippy::too_many_arguments)]
fn coop_fill_cone(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    cone: &Cone,
    base: OpId,
    sr: Dim,
    sc: Dim,
    rows: Dim,
    chunk: Dim,
    threads: Dim,
    swizzle: bool,
) -> (OpId, OpId) {
    coop_fill_load_with(kernel, before, lane, OpId::NULL, Some(cone), base, sr, sc, rows, chunk, threads, swizzle)
}

#[allow(clippy::too_many_arguments)]
fn coop_fill_load_with(
    kernel: &mut Kernel,
    before: OpId,
    lane: OpId,
    src: OpId,
    cone: Option<&Cone>,
    base: OpId,
    sr: Dim,
    sc: Dim,
    rows: Dim,
    chunk: Dim,
    threads: Dim,
    swizzle: bool,
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
    // The element's two coordinates, handed to the cone instead of the linear
    // index it would have to divide them back out of: the row/column is the
    // thread's row plus its step, the contraction element the vector's start —
    // plus the chunk's K origin, which the tile-local column does not carry.
    let row = put(kernel, Op::Binary { x: tid_row, y: rrow, bop: BOp::Add });
    let mut coord_k = put(kernel, Op::Binary { x: cterm, y: vector, bop: BOp::Add });
    let mut row = row;
    if let Some(cone) = cone {
        let ke = put(kernel, Op::Const(Constant::idx(cone.k_extent)));
        let koff = put(kernel, Op::Binary { x: base, y: ke, bop: BOp::Mod });
        let roff = put(kernel, Op::Binary { x: base, y: ke, bop: BOp::Div });
        coord_k = put(kernel, Op::Binary { x: coord_k, y: koff, bop: BOp::Add });
        row = put(kernel, Op::Binary { x: row, y: roff, bop: BOp::Add });
    }
    let value = match cone {
        Some(cone) => cone.clone_at_coords(kernel, before, row, coord_k),
        None => {
            let gidx = put(kernel, Op::Binary { x: base, y: rterm_v, bop: BOp::Add });
            let gidx = put(kernel, Op::Binary { x: gidx, y: cterm, bop: BOp::Add });
            let gidx = put(kernel, Op::Binary { x: gidx, y: rterm_s, bop: BOp::Add });
            let gidx = put(kernel, Op::Binary { x: gidx, y: vector, bop: BOp::Add });
            put(kernel, Op::Load { src, src_idx: gidx })
        }
    };
    // The shared slot is the k-block-major tile layout —
    // `((k/8)*(rows/8) + (m/8))*64 + (m%8)*8 + (k%8)` — so a vector is one
    // eight-half row of a k-block and its lanes are contiguous. Every other
    // term is a shift or a mask, and the tile packs to `rows * chunk` halves
    // with no padding: the reference's 16 KB per operand, and the 32 KB that
    // keeps two blocks resident per SM.
    let three = put(kernel, Op::Const(Constant::idx(LDMATRIX_LOG2)));
    let seven = put(kernel, Op::Const(Constant::idx(LDMATRIX_EDGE - 1)));
    let mblk = put(kernel, Op::Binary { x: row, y: three, bop: BOp::BitShiftRight });
    let kblk = put(kernel, Op::Binary { x: cterm, y: three, bop: BOp::BitShiftRight });
    // Row swizzle: the eight lanes of a store share a row of their 128-byte
    // block and differ only in their k-block, so with a linear index every one
    // of them lands on banks 0-3 (measured: 8-way conflict on each of the 16
    // fill stores, `mio_throttle` the top stall). XORing the row field with the
    // unit's k-block spreads them over the banks, and `coop_ldmatrix` mirrors it
    // with the k-block of the unit it reads. The plain layout is the same pack
    // without the XOR: the two are the staged layout's axis, and every read of
    // the tile takes the flag of the fill that wrote it.
    let mrow = put(kernel, Op::Binary { x: row, y: seven, bop: BOp::BitAnd });
    let mrow = if swizzle {
        let swz = put(kernel, Op::Binary { x: kblk, y: seven, bop: BOp::BitAnd });
        put(kernel, Op::Binary { x: mrow, y: swz, bop: BOp::BitXor })
    } else {
        mrow
    };
    let mrow = put(kernel, Op::Binary { x: mrow, y: three, bop: BOp::BitShiftLeft });
    // The column's offset inside its eight-half block: the slot is
    // `... + (k%8)`, and `vector` only covers the vector's own lanes. Without
    // this term every thread of a row writes the same eight halves, so a tile
    // holds one vector per row instead of the whole window.
    let kin = put(kernel, Op::Binary { x: cterm, y: seven, bop: BOp::BitAnd });
    let rblk = put(kernel, Op::Const(Constant::idx(rows / LDMATRIX_EDGE)));
    let kblk = put(kernel, Op::Binary { x: kblk, y: rblk, bop: BOp::Mul });
    let blk = put(kernel, Op::Binary { x: kblk, y: mblk, bop: BOp::Add });
    let pack = put(kernel, Op::Const(Constant::idx(LDMATRIX_BLOCK)));
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
/// Stage one operand tile in the packed-word layout the `ldmatrix` mainloop
/// reads: the payload words at `payload` and the per-group scales at
/// `scale_col`, both inside a row of `stride` words. The code's word is loaded
/// once per `codes` elements and expanded into `codes / 4` operand words — a
/// copy when the loaded element's bytes are the codes themselves
/// ([`Quant::byte_word`]), otherwise the two-bit unpack, whose lookup table is
/// the code's own centering.
/// Stage one operand tile in the packed-word layout the `ldmatrix` mainloop
/// reads: the payload words at `payload` and the per-group scales at
/// `scale_col`, both inside a row of `stride` words.
///
/// The payload half has two shapes, chosen by what the cone offers: when the
/// quantization packs a code into one loadable word (`word`), that word is
/// loaded once per `codes` elements and expanded into `codes / 4` operand words
/// — a copy when the loaded element's bytes are the codes themselves
/// ([`Quant::byte_word`]), otherwise the two-bit unpack, whose lookup table is
/// the code's own centering. When the cone has no such word, the four codes of
/// an operand word are read one by one and packed by hand. Either way the tile,
/// the row stride and the scale columns are the same: the staged layout does not
/// depend on how the codes are fetched.
/// The 32-bit operand word at `(row, k)`: four consecutive codes packed by hand,
/// the byte an `m8n8k16` lane multiplies. `clone_at_coords` returns one element's
/// code whatever the format, so a byte-addressed code and a packed multi-bit one
/// build the same word — this is the form for a cone with no loadable code word,
/// and the form every unstaged fragment read takes.
fn cone_word4(kernel: &mut Kernel, before: OpId, quant: &Quant, row: OpId, k: OpId) -> OpId {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    let mask = put(kernel, Op::Const(Constant::new(0xffu32)));
    let mut packed = OpId::NULL;
    for b in 0..4u32 {
        let c = put(kernel, Op::Const(Constant::idx(i64::from(b))));
        let c = put(kernel, Op::Binary { x: k, y: c, bop: BOp::Add });
        let v = quant.code.clone_at_coords(kernel, before, row, c);
        let v = if kernel.dtype(v) == DType::I32 {
            v
        } else {
            put(kernel, Op::Cast { x: v, dtype: DType::I32 })
        };
        let v = put(kernel, Op::Cast { x: v, dtype: DType::U32 });
        let v = put(kernel, Op::Binary { x: v, y: mask, bop: BOp::BitAnd });
        let sh = put(kernel, Op::Const(Constant::new(8 * b)));
        let v = put(kernel, Op::Binary { x: v, y: sh, bop: BOp::BitShiftLeft });
        packed = if b == 0 {
            v
        } else {
            put(kernel, Op::Binary { x: packed, y: v, bop: BOp::Add })
        };
    }
    packed
}

#[allow(clippy::too_many_arguments)]
fn coop_fill_tile(
    kernel: &mut Kernel,
    before: OpId,
    tid: OpId,
    quant: &Quant,
    word: Option<&CodeWord>,
    row_off: OpId,
    k_off: OpId,
    rows: Dim,
    chunk: Dim,
    threads: Dim,
    tile: OpId,
    stride: Dim,
    payload: Dim,
    scale_col: Dim,
    axis: u32,
) {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    // ---- the payload words ----
    // A fetched unit is one code word (`codes` elements) and covers
    // `codes / 4` operand words; without a code word a unit is one operand word.
    let unit = word.map_or(4, |w| w.codes);
    let per_row = chunk / Dim::from(unit);
    let total = rows * per_row;
    let steps = (total + threads - 1) / threads;
    let step = put(kernel, Op::Unroll { axis, len: u16::try_from(steps).expect("word fill step count") });
    let th = put(kernel, Op::Const(Constant::idx(threads)));
    let st = put(kernel, Op::Binary { x: step, y: th, bop: BOp::Mul });
    let w = put(kernel, Op::Binary { x: tid, y: st, bop: BOp::Add });
    // A tile smaller than the block leaves threads past its end: the last step
    // wraps instead of overrunning, writing the same value at the same address
    // (a repeat, not a race).
    let w = if total % threads != 0 {
        let ct = put(kernel, Op::Const(Constant::idx(total)));
        put(kernel, Op::Binary { x: w, y: ct, bop: BOp::Mod })
    } else {
        w
    };
    let prc = put(kernel, Op::Const(Constant::idx(per_row)));
    let r = put(kernel, Op::Binary { x: w, y: prc, bop: BOp::Div });
    let c = put(kernel, Op::Binary { x: w, y: prc, bop: BOp::Mod });
    let cc = put(kernel, Op::Const(Constant::idx(Dim::from(unit))));
    let ck = put(kernel, Op::Binary { x: c, y: cc, bop: BOp::Mul });
    let row = put(kernel, Op::Binary { x: r, y: row_off, bop: BOp::Add });
    let kk = put(kernel, Op::Binary { x: ck, y: k_off, bop: BOp::Add });
    let out_words = Dim::from(unit / 4);
    let ow = put(kernel, Op::Const(Constant::idx(out_words)));
    let cw = put(kernel, Op::Binary { x: c, y: ow, bop: BOp::Mul });
    let cpay = put(kernel, Op::Const(Constant::idx(payload)));
    let cw = put(kernel, Op::Binary { x: cw, y: cpay, bop: BOp::Add });
    let cst = put(kernel, Op::Const(Constant::idx(stride)));
    let mut base = put(kernel, Op::Binary { x: r, y: cst, bop: BOp::Mul });
    base = put(kernel, Op::Binary { x: base, y: cw, bop: BOp::Add });
    let Some(word) = word else {
        // No code word: pack the four codes of this operand word by hand.
        let packed = cone_word4(kernel, before, quant, row, kk);
        put(kernel, Op::Store { dst: tile, dst_idx: base, src: packed });
        return coop_fill_scales(kernel, before, tid, quant, row_off, k_off, rows, chunk, threads, tile, stride, scale_col, axis);
    };
    // The cone's own word load at this element: the fill stages the packed
    // operand without knowing the format, only its width.
    let q = quant.code.clone_op_at_coords(kernel, before, row, kk, word.load);
    if quant.byte_word(word) {
        put(kernel, Op::Store { dst: tile, dst_idx: base, src: q });
    } else {
        assert!(
            word.bits == 2 && word.codes == 8 && word.dtype == DType::U16,
            "the packed word fill unpacks two-bit codes, eight to a sixteen-bit word"
        );
        // The reference reads the half signed and turns each pair of codes
        // into a byte with the table `i - zero`, the code's centering.
        let qs = put(kernel, Op::Bitcast { x: q, dtype: DType::I16 });
        let qi = put(kernel, Op::Cast { x: qs, dtype: DType::I32 });
        let qu = put(kernel, Op::Bitcast { x: qi, dtype: DType::U32 });
        let mut table = 0u32;
        for i in 0..4i32 {
            table |= u32::from((i - word.zero) as u8) << (8 * i as u32);
        }
        let tb = put(kernel, Op::Const(Constant::new(table)));
        let two = put(kernel, Op::Const(Constant::new(2u32)));
        let mask = put(kernel, Op::Const(Constant::new(0x7777u32)));
        let sel_even = put(kernel, Op::Const(Constant::new(0x5140u32)));
        let sel_odd = put(kernel, Op::Const(Constant::new(0x7362u32)));
        let even = put(kernel, Op::Binary { x: qu, y: mask, bop: BOp::BitAnd });
        let qe = put(kernel, Op::PTX(Box::new(PTXOp::Prmt { a: tb, b: tb, c: even })));
        let sh = put(kernel, Op::Binary { x: qu, y: two, bop: BOp::BitShiftRight });
        let odd = put(kernel, Op::Binary { x: sh, y: mask, bop: BOp::BitAnd });
        let qo = put(kernel, Op::PTX(Box::new(PTXOp::Prmt { a: tb, b: tb, c: odd })));
        let qx = put(kernel, Op::PTX(Box::new(PTXOp::Prmt { a: qe, b: qo, c: sel_even })));
        let qy = put(kernel, Op::PTX(Box::new(PTXOp::Prmt { a: qe, b: qo, c: sel_odd })));
        put(kernel, Op::Store { dst: tile, dst_idx: base, src: qx });
        let one = put(kernel, Op::Const(Constant::idx(1i64)));
        let base1 = put(kernel, Op::Binary { x: base, y: one, bop: BOp::Add });
        put(kernel, Op::Store { dst: tile, dst_idx: base1, src: qy });
    }
    // ---- the per-group scales ----
    coop_fill_scales(kernel, before, tid, quant, row_off, k_off, rows, chunk, threads, tile, stride, scale_col, axis);
}

/// The scale half of a staged packed tile: one f32 per (row, group), stored at
/// `row * stride + scale_col + group`. Every operand of the packed layout keeps
/// its scales inside its own row, so the mainloop reads them with the payload.
#[allow(clippy::too_many_arguments)]
fn coop_fill_scales(
    kernel: &mut Kernel,
    before: OpId,
    tid: OpId,
    quant: &Quant,
    row_off: OpId,
    k_off: OpId,
    rows: Dim,
    chunk: Dim,
    threads: Dim,
    tile: OpId,
    stride: Dim,
    scale_col: Dim,
    axis: u32,
) {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    let cst = put(kernel, Op::Const(Constant::idx(stride)));
    let groups = chunk / Q8_1_GROUP;
    let stotal = rows * groups;
    let ssteps = (stotal + threads - 1) / threads;
    let sstep = put(kernel, Op::Unroll { axis, len: u16::try_from(ssteps).expect("scale fill step count") });
    let sth = put(kernel, Op::Const(Constant::idx(threads)));
    let sst = put(kernel, Op::Binary { x: sstep, y: sth, bop: BOp::Mul });
    let e = put(kernel, Op::Binary { x: tid, y: sst, bop: BOp::Add });
    let e = if stotal % threads != 0 {
        let ct = put(kernel, Op::Const(Constant::idx(stotal)));
        put(kernel, Op::Binary { x: e, y: ct, bop: BOp::Mod })
    } else {
        e
    };
    let gc = put(kernel, Op::Const(Constant::idx(groups)));
    let sr = put(kernel, Op::Binary { x: e, y: gc, bop: BOp::Div });
    let sg = put(kernel, Op::Binary { x: e, y: gc, bop: BOp::Mod });
    let kg = put(kernel, Op::Const(Constant::idx(Q8_1_GROUP)));
    let ko = put(kernel, Op::Binary { x: sg, y: kg, bop: BOp::Mul });
    let row = put(kernel, Op::Binary { x: sr, y: row_off, bop: BOp::Add });
    let kk = put(kernel, Op::Binary { x: ko, y: k_off, bop: BOp::Add });
    let val = quant.scale.clone_at_coords(kernel, before, row, kk);
    let val = if kernel.dtype(val) == DType::F32 {
        val
    } else {
        put(kernel, Op::Cast { x: val, dtype: DType::F32 })
    };
    let bits = put(kernel, Op::Bitcast { x: val, dtype: DType::U32 });
    let csc = put(kernel, Op::Const(Constant::idx(scale_col)));
    let mut at = put(kernel, Op::Binary { x: sr, y: cst, bop: BOp::Mul });
    at = put(kernel, Op::Binary { x: at, y: csc, bop: BOp::Add });
    at = put(kernel, Op::Binary { x: at, y: sg, bop: BOp::Add });
    put(kernel, Op::Store { dst: tile, dst_idx: at, src: bits });
}

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
    let three = put(kernel, Op::Const(Constant::idx(LDMATRIX_LOG2)));
    let seven = put(kernel, Op::Const(Constant::idx(LDMATRIX_EDGE - 1)));
    let mblk = put(kernel, Op::Binary { x: row, y: three, bop: BOp::BitShiftRight });
    let mrow = put(kernel, Op::Binary { x: row, y: seven, bop: BOp::BitAnd });
    let mrow = put(kernel, Op::Binary { x: mrow, y: three, bop: BOp::BitShiftLeft });
    let pack = put(kernel, Op::Const(Constant::idx(LDMATRIX_BLOCK)));
    let mut addr = put(kernel, Op::Binary { x: mblk, y: pack, bop: BOp::Mul });
    addr = put(kernel, Op::Binary { x: addr, y: mrow, bop: BOp::Add });
    if k / LDMATRIX_EDGE > 1 {
        let seg = put(kernel, Op::Binary { x: lane, y: rconst, bop: BOp::Div });
        let kstride = put(kernel, Op::Const(Constant::idx(tile_rows / LDMATRIX_EDGE * LDMATRIX_BLOCK)));
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
    swizzle: bool,
) -> OpId {
    let put = |kernel: &mut Kernel, op: Op| -> OpId {
        if before.is_null() {
            kernel.push_back(op)
        } else {
            kernel.insert_before(before, op)
        }
    };
    let frags = u8::try_from(rows / LDMATRIX_EDGE * (k / LDMATRIX_EDGE)).expect("ldmatrix: too many fragments");
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
    let off = put(kernel, Op::Const(Constant::idx(k_offset * (tile_rows / LDMATRIX_EDGE * LDMATRIX_EDGE))));
    let addr = put(kernel, Op::Binary { x: addr, y: off, bop: BOp::Add });
    // Row swizzle, the read half. The fill XORs the row field with the k-block
    // it wrote, so the read must address the row the element actually lives in:
    // XOR the same field (8 halves per row, bits 3..5 of the element index) with
    // the k-block of the unit being read. Reading it back off the index itself
    // (`index / halves per k-block`) is what keeps this identical to the fill's
    // key without either side restating the other's lane geometry — and since
    // every shared address here is loop-invariant, it is computed once, not per
    // iteration.
    // Row swizzle, the read half. The fill XORs the row field with the k-block
    // it wrote, so the read must address the row the element actually lives in:
    // XOR the same field (8 halves per row, bits 3..5 of the element index) with
    // the k-block of the unit being read. Reading it back off the index itself
    // (`index / halves per k-block`) is what keeps this identical to the fill's
    // key without either side restating the other's lane geometry — and since
    // every shared address here is loop-invariant, it is computed once, not per
    // iteration. A plain (unswizzled) tile was written row-linearly, so the
    // address the lane base gives is already the slot the fill wrote.
    let addr = if swizzle {
        let one = put(kernel, Op::Const(Constant::idx(1i64)));
        let eight = put(kernel, Op::Const(Constant::idx(LDMATRIX_EDGE)));
        let kbh = put(kernel, Op::Const(Constant::idx(tile_rows / LDMATRIX_EDGE * LDMATRIX_BLOCK)));
        let key = put(kernel, Op::Binary { x: addr, y: kbh, bop: BOp::Div });
        // The key must be the k-block *inside the operand's tile*, the way the
        // fill writes it: the absolute address also carries the tile's base, so
        // the key is folded to the same width as the store's. The fill XORs the
        // low *three* bits of the k-block into the row field (bits 3..5 of the
        // element index); a tile with more than eight k-blocks therefore has to
        // fold the read key the same way, or the XOR reaches bit 6 and names the
        // wrong k-block.
        let nkb = put(kernel, Op::Const(Constant::idx(kblocks.clamp(1, LDMATRIX_EDGE))));
        let kmask = put(kernel, Op::Binary { x: nkb, y: one, bop: BOp::Sub });
        let key = put(kernel, Op::Binary { x: key, y: kmask, bop: BOp::BitAnd });
        let swz = put(kernel, Op::Binary { x: key, y: eight, bop: BOp::Mul });
        put(kernel, Op::Binary { x: addr, y: swz, bop: BOp::BitXor })
    } else {
        addr
    };
    put(
        kernel,
        Op::PTX(Box::new(PTXOp::VLdMatrix {
            dims: LdMatrixDims::m8n8,
            num: LdMatrixNum::from_words(frags),
            trans: LdMatrixTrans::off,
            dtype: LdMatrixType::b16,
            src: tile,
            addr,
        })),
    )
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

/// A tile shape as one plan asks the tensor core for it: a grid of
/// `mt x nt x kt` instructions, the instruction's own `m x n x k` extent, and
/// the K step each operand's view already covers (`a_k`/`b_k` are 1 for a plain
/// K-contiguous operand).
#[derive(Clone, Copy)]
pub(crate) struct TileShape {
    mt: Dim,
    nt: Dim,
    kt: Dim,
    fm: Dim,
    fn_: Dim,
    fk: Dim,
    a_k: Dim,
    b_k: Dim,
    slots: Dim,
}

/// The form a [`TileShape`] takes at one warp layout and staging request: how
/// many warps own which band of the tile grid, the register block one warp
/// owns, and whether the operands go through shared memory. `None` when the
/// request cannot be emitted as one kernel.
#[derive(Clone, Copy)]
struct TileFit {
    mt_w: Dim,
    nt_w: Dim,
    warps: u32,
    combine: bool,
    staged: bool,
    chunk: Dim,
    interleave: bool,
}

/// Whether a tile shape can be emitted at this warp layout, or `None` if it
/// cannot: a warp grid that does not divide the tile grid, a K split that does
/// not divide the loop, a combine epilogue that cannot store warp-wide, a
/// staged form whose tile or combine buffer does not fit shared memory, or an
/// unstage register block larger than the register file. The enumeration only
/// offers shapes this accepts, and [`tile_to_wmma`] panics on anything else,
/// so a requested form is always the emitted form — never a fallback, never a
/// skip.
fn tile_fit(
    shape: &TileShape,
    staged: bool,
    layout: (u16, u16, u16),
    info: &DeviceInfo,
    k_divides: impl Fn(u16) -> bool,
) -> Option<TileFit> {
    let TileShape { mt, nt, kt, fm, fn_, fk, a_k, b_k, slots } = *shape;
    let warp = u32::from(info.warp_size);
    let chunk = kt * fk;
    // The block-wide form stages a window of two K steps (one `ldmatrix` per
    // row group covering both) in two buffers, so the output tile is not
    // limited by the whole K chunk living in shared memory.
    // The staged window: the two K steps one `ldmatrix` covers, in two
    // buffers. A wider window costs shared memory and measured slower.
    let window_width = if kt % 2 == 0 { fk * 2 } else { fk };
    // The block's warp layout: `m` warps over the output rows and `n` over the
    // output columns (a warp grid — every warp owns its sub-tile, so nothing
    // is summed across warps), or `k` warps over the contraction (the partial
    // accumulators are summed through shared memory). One axis at a time.
    let (wm, wn, wk) = layout;
    let cols = nt * fn_;
    let rows = if fn_ > fm { fn_ } else { fm };
    let staged_form = staged && a_k == 1 && b_k == 1;
    // The staged form holds one warp-private tile plus, for a K split, the
    // combine buffer `[output column][warp * rows + row]`, which needs the
    // block's output rows (`nt * fn_`) to be whole warp-wide stores.
    let combine = wk > 1;
    if combine && cols % Dim::from(warp) != 0 {
        return None;
    }
    let staged_bytes = |nw: Dim| {
        // The block-wide form holds two stages of a two-K-step window; a
        // split-K warp holds its own tile and merges through the combine
        // buffer.
        let tile = if combine {
            nw * rows * chunk * 2
        } else {
            (mt * fm + nt * fn_) * chunk * 2
        };
        if combine { tile + mt * fm * (nw * cols + 4) * 4 } else { tile }
    };
    // A warp grid must divide the tile grid, and a K-split warp count must
    // divide the loop. The loop length is a symbolic expression: its
    // divisibility is decided on the expression, and only the K-split warp
    // count needs a value at all.
    if (wk > 1 && !k_divides(wk)) || (wm > 1 && mt % Dim::from(wm) != 0) || (wn > 1 && nt % Dim::from(wn) != 0) {
        return None;
    }
    // The tiles one warp owns: a warp grid hands each warp a contiguous band
    // of the block's tile grid.
    let mt_w = mt / Dim::from(if wm > 1 { wm } else { 1 });
    let nt_w = nt / Dim::from(if wn > 1 { wn } else { 1 });
    let warps = u32::from(wm) * u32::from(wn) * u32::from(wk);
    let nw = Dim::from(warps);
    // A block-wide fill also needs the tile to hold at least one vector per
    // thread and to divide evenly among them; the staged form's `ldmatrix`
    // group is one lane over both operands and every K pair, which needs the
    // operands' loads to share shape and stride: equal tile rows and half as
    // many m tiles per warp as n tiles. A split-K warp owns a private tile and
    // never forms the block-wide group.
    let threads = Dim::from(warp);
    let vecs = |rows: Dim| rows * (window_width / Dim::from(2));
    let all = nw * threads;
    let fill_fits =
        combine || (vecs(mt * fm) >= all && vecs(mt * fm) % all == 0) && (vecs(nt * fn_) >= all && vecs(nt * fn_) % all == 0);
    let lane_group = combine || (mt * fm == nt * fn_ && mt_w * Dim::from(2) == nt_w);
    let staged = staged_form && lane_group && fill_fits && staged_bytes(nw) <= info.local_mem_size;
    // The requested form must be the emitted form: a staged request that the
    // gates above turn into the direct form is a different kernel, not this
    // plan's.
    if staged != staged_form {
        return None;
    }
    // A thread's accumulators must fit the register file, and the unstage form
    // emits one body per tile grid entry: a grid too wide for the registers or
    // the IR is not a kernel.
    if !staged && (mt_w * nt_w * slots > Dim::from(128) || mt * nt * kt > Dim::from(256)) {
        return None;
    }
    // The staged form slices K round-robin: warp `y` takes chunks `y`,
    // `y + n`, ... of one tile step each, so the warps' loads of a row stay
    // in one window of it. A contiguous slice otherwise.
    let interleave = staged_form && combine;
    Some(TileFit { mt_w, nt_w, warps, combine, staged, chunk, interleave })
}

/// Whether the staged element form's fragment feed is the instruction's.
///
/// The `ldmatrix` a staged step issues covers `rows / 8 * (k / 8)` 32-bit words
/// of one operand, and the instruction takes `a.slots` / `b.slots` *elements*
/// per lane, `32 / dtype.bit_size()` of them to a word. A plan whose two counts
/// differ hands the `mma` an operand list that is not the instruction's: the
/// device rejects the PTX, and the slice that would cut the extra words out of
/// a missing one indexes a scalar. So the identity is a candidate gate, at both
/// step widths the emitter uses (one `ldmatrix` per step, or one for a pair).
fn fragment_width_fits(v: &MMAVariant, shape: &TileShape) -> bool {
    let per_a = usize::from(32 / v.a_dtype.bit_size());
    let per_b = usize::from(32 / v.b_dtype.bit_size());
    let (a_slots, b_slots) = (v.a.slots.len(), v.b.slots.len());
    if a_slots % per_a != 0 || b_slots % per_b != 0 {
        return false;
    }
    let (a_words, b_words) = (a_slots / per_a, b_slots / per_b);
    if a_words == 0 || b_words == 0 {
        return false;
    }
    let (a_words, b_words) = (Dim::from(a_words as i64), Dim::from(b_words as i64));
    let words = |rows: Dim, k: Dim| rows / LDMATRIX_EDGE * (k / LDMATRIX_EDGE);
    let (fm, fn_, fk) = (shape.fm, shape.fn_, shape.fk);
    // One step always; the pair (the emitter's `fk * 2` window) only exists
    // when the K step count is even and greater than one.
    let single = words(fm, fk) == a_words && words(fn_, fk) == b_words;
    let paired = shape.kt < Dim::from(2)
        || (words(fm, fk * Dim::from(2)) == a_words * 2 && words(fn_, fk * Dim::from(2)) == b_words * 2);
    single && paired
}

/// Whether the int8 form can be emitted at this shape, staging level and
/// layout. The instruction's own geometry (`m8n8k16` for int8, one
/// `ldmatrix.x4` fragment pair per K group) is fixed; the tile grid, the K chunk
/// and the warp grid are the shape's.
///
/// A staged form fills two packed tiles and reads them back: the tile grid, the
/// K chunk and the warp grid must divide the fills over the block's threads
/// exactly, the fragment lanes must fit the register file, and the two tiles the
/// shared memory. An unstaged form has no tiles and no fill: it reads each
/// fragment word straight from the operand's cone, so it is offered only where
/// the per-lane word is cheap — a cone whose code is a field of one loaded word.
fn s8_fit(shape: &TileShape, staged: bool, swizzle: bool, layout: (u16, u16, u16), info: &DeviceInfo) -> bool {
    let TileShape { mt, nt, kt, fm, fn_, fk, a_k, b_k, slots } = *shape;
    let (wm, wn, wk) = layout;
    // One warp per output band, no split-K: the sums are scaled per group and
    // a K split would have to sum scaled integer partials.
    if wk > 1 || a_k != 1 || b_k != 1 {
        return false;
    }
    if fm != Dim::from(8) || fn_ != Dim::from(8) || fk != Dim::from(16) {
        return false;
    }
    let wm = Dim::from(wm.max(1));
    let wn = Dim::from(wn.max(1));
    if mt % wm != 0 || nt % wn != 0 {
        return false;
    }
    let mt_w = mt / wm;
    let nt_w = nt / wn;
    // One `ldmatrix.x4` covers two n-fragments of the A tile.
    if mt_w % Dim::from(2) != 0 {
        return false;
    }
    let chunk = kt * fk;
    let groups = chunk / Q8_1_GROUP;
    if groups == 0 || chunk % Q8_1_GROUP != 0 {
        return false;
    }
    // The fragment lane's registers.
    if mt_w * nt_w * slots > Dim::from(128) {
        return false;
    }
    if !staged {
        // Without a tile the lane builds each fragment word from four code
        // reads, which every format can do (a packed code word is not needed).
        return true;
    }
    let rows_a = mt * fm;
    let rows_b = nt * fn_;
    let payload = chunk / Dim::from(4);
    let xs = packed_row_stride(payload, groups, swizzle);
    let ys = groups + payload;
    if (rows_a * xs + rows_b * ys) * Dim::from(4) > Dim::from(info.local_mem_size) {
        return false;
    }
    true
}

/// The A row's stride in 32-bit words for the packed form: the payload words,
/// then the group scales. The swizzled layout pads that up to a 16-byte boundary
/// (every `ldmatrix` lane base must be aligned) and one four-word tail so the
/// rows of one `ldmatrix` land on different bank sets; the plain layout is just
/// the aligned row, which the `ldmatrix` reads as a linear pack.
fn packed_row_stride(payload: Dim, groups: Dim, swizzle: bool) -> Dim {
    let used = payload + groups;
    let aligned = (used + Dim::from(3)) / Dim::from(4) * Dim::from(4);
    if swizzle { aligned + Dim::from(4) } else { aligned }
}

/// Emit the int8 form: both operands' codes staged as their packed words — the
/// `mmq` shape — read back by the fragment width the instruction consumes. A's
/// row holds its payload words first, then the group scales, then (swizzled) a
/// four-word tail; B's row holds the scales first and the payload words after
/// them. A is read with one `ldmatrix.x4.b16` per (two-fragment, K group) pair,
/// B with two words per fragment, and the four int32 sums of a fragment pair
/// are folded into the f32 accumulator once per group.
///
/// Both operands use the same staging routine ([`coop_fill_tile`]): the code's
/// word is loaded once per `codes` elements when the cone packs its codes into
/// one loaded word, and packed element by element when it does not, so a
/// two-bit weight block, an already-int8 activation block and a cone with no
/// code word all cost one staging pass into the same tile.
///
/// An unstaged form (`staged == false`) skips the tiles entirely: each
/// fragment's word is read straight from its operand's cone, which is why only a
/// cone with a code word is offered there. Everything after the fragment feed —
/// the instruction chain, the per-group scale, the write-back — is the same
/// body for both, which is what makes the staging an axis of this lowering
/// rather than a second lowering.
fn tile_to_wmma_s8_cell(kernel: &mut Kernel, t: &Tiled, staged: bool, swizzle: bool, layout: (u16, u16, u16)) {
    let v = &t.variant;
    let (fm, fn_, fk) = {
        let (m, n, k) = v.dims.decompose_mnk();
        (m as Dim, n as Dim, k as Dim)
    };
    let info = kernel.device_info();
    let warp = u32::from(info.warp_size);
    let (wm, wn, _) = layout;
    let warps = u32::from(wm.max(1)) * u32::from(wn.max(1));
    let mt_w = t.mt / Dim::from(wm.max(1));
    let nt_w = t.nt / Dim::from(wn.max(1));
    let slots = t.slots;
    let chunk = t.kt * fk;
    let groups = chunk / Q8_1_GROUP;
    // The row of each staged tile, in 32-bit words: A's payload, A's scales,
    // (swizzled) a four-word tail that keeps the rows of one `ldmatrix` off one
    // bank set; B's scales, then B's payload.
    let payload = chunk / 4;
    let xs = packed_row_stride(payload, groups, swizzle);
    let ys = groups + payload;
    let (Some(qa), Some(qb)) = (t.a_quant.as_ref(), t.b_quant.as_ref()) else {
        panic!("the int8 form needs both operands split into a code and a group scale")
    };
    let wa = qa.code_word(kernel);
    let wb = qb.code_word(kernel);
    let rows_a = t.mt * fm;
    let rows_b = t.nt * fn_;
    let (xt, yt) = if staged {
        (
            insert_scoped_storage(kernel, DType::U32, MemScope::Local, rows_a * xs),
            insert_scoped_storage(kernel, DType::U32, MemScope::Local, rows_b * ys),
        )
    } else {
        (OpId::NULL, OpId::NULL)
    };

    // ---- the accumulator: one f32 sum per (fragment, C word) ----
    let len = mt_w * nt_w * slots;
    if let Op::Storage { len: l, .. } = &mut kernel.ops[t.acc].op {
        *l = len;
    }
    let zero_f = match kernel.at(t.init) {
        Op::Store { src, .. } => *src,
        _ => unreachable!("tiled tensorize: init is not a store"),
    };
    let clear_lane = kernel.insert_before(t.init, Op::Unroll { axis: 0, len: u16::try_from(len).expect("accumulator words") });
    kernel.insert_before(t.init, Op::Store { dst: t.acc, dst_idx: clear_lane, src: zero_f });
    kernel.remove_op(t.init);

    // ---- the warp's place in the block ----
    let put = |kernel: &mut Kernel, op: Op| -> OpId { kernel.insert_before(t.loop_id, op) };
    let lane = put(kernel, Op::Range { axis: 0, kind: RangeKind::Local(warp) });
    let mut warp_var = OpId::NULL;
    if warps > 1 {
        warp_var = put(kernel, Op::Range { axis: 1, kind: RangeKind::Local(warps) });
    }
    let tid = if warp_var.is_null() {
        lane
    } else {
        let w = put(kernel, Op::Const(Constant::idx(Dim::from(warp))));
        let base = put(kernel, Op::Binary { x: warp_var, y: w, bop: BOp::Mul });
        put(kernel, Op::Binary { x: base, y: lane, bop: BOp::Add })
    };
    let band_m = if wn > 1 {
        let c = put(kernel, Op::Const(Constant::idx(Dim::from(wn))));
        put(kernel, Op::Binary { x: warp_var, y: c, bop: BOp::Div })
    } else if wm > 1 {
        warp_var
    } else {
        OpId::NULL
    };
    let band_n = if wn > 1 {
        let c = put(kernel, Op::Const(Constant::idx(Dim::from(wn))));
        put(kernel, Op::Binary { x: warp_var, y: c, bop: BOp::Mod })
    } else {
        OpId::NULL
    };
    let warp_off_m = if band_m.is_null() {
        OpId::NULL
    } else {
        let s = put(kernel, Op::Const(Constant::idx(mt_w * fm)));
        put(kernel, Op::Binary { x: band_m, y: s, bop: BOp::Mul })
    };
    let warp_off_n = if band_n.is_null() {
        OpId::NULL
    } else {
        let s = put(kernel, Op::Const(Constant::idx(nt_w * fn_)));
        put(kernel, Op::Binary { x: band_n, y: s, bop: BOp::Mul })
    };
    // A warp grid of one warp along an axis has no band offset: the band term
    // is the identity, not a null operand.
    let add_off = |kernel: &mut Kernel, base: OpId, off: OpId| -> OpId {
        if off.is_null() {
            base
        } else {
            kernel.insert_before(t.update, Op::Binary { x: base, y: off, bop: BOp::Add })
        }
    };
    // The lane's fragment geometry: `gid = lane/4` is its row, `tig = lane%4`
    // the four-int8 K group; the `ldmatrix` address is `lane%16` rows by
    // `lane/16` 16-byte halves.
    let four = put(kernel, Op::Const(Constant::idx(4i64)));
    let gid = put(kernel, Op::Binary { x: lane, y: four, bop: BOp::Div });
    let tig = put(kernel, Op::Binary { x: lane, y: four, bop: BOp::Mod });
    let two = put(kernel, Op::Const(Constant::idx(2i64)));
    let tig2 = put(kernel, Op::Binary { x: tig, y: two, bop: BOp::Mul });
    let c16 = put(kernel, Op::Const(Constant::idx(2 * fm)));
    let lanerow = put(kernel, Op::Binary { x: lane, y: c16, bop: BOp::Mod });
    let lanehalf = put(kernel, Op::Binary { x: lane, y: c16, bop: BOp::Div });

    // ---- stage both tiles for this K chunk ----
    let end = t.update;
    let a_base = base_expr(kernel, end, &t.a_fixed, t.a_offset, 0);
    let b_base = base_expr(kernel, end, &t.b_fixed, t.b_offset, 0);
    let a_roff = {
        let ke = kernel.insert_const_idx_before(end, qa.code.k_extent);
        kernel.insert_before(end, Op::Binary { x: a_base, y: ke, bop: BOp::Div })
    };
    let a_koff = {
        let ke = kernel.insert_const_idx_before(end, qa.code.k_extent);
        kernel.insert_before(end, Op::Binary { x: a_base, y: ke, bop: BOp::Mod })
    };
    let b_roff = {
        let ke = kernel.insert_const_idx_before(end, qb.code.k_extent);
        kernel.insert_before(end, Op::Binary { x: b_base, y: ke, bop: BOp::Div })
    };
    let b_koff = {
        let ke = kernel.insert_const_idx_before(end, qb.code.k_extent);
        kernel.insert_before(end, Op::Binary { x: b_base, y: ke, bop: BOp::Mod })
    };
    let threads = Dim::from(warps) * Dim::from(warp);
    let cxs = kernel.insert_before(end, Op::Const(Constant::idx(xs)));
    let cys = kernel.insert_before(end, Op::Const(Constant::idx(ys)));
    let cgrp = kernel.insert_before(end, Op::Const(Constant::idx(groups)));
    let cpay = kernel.insert_before(end, Op::Const(Constant::idx(payload)));
    if staged {
        coop_fill_tile(kernel, end, tid, qa, wa.as_ref(), a_roff, a_koff, rows_a, chunk, threads, xt, xs, 0, payload, 2);
        coop_fill_tile(kernel, end, tid, qb, wb.as_ref(), b_roff, b_koff, rows_b, chunk, threads, yt, ys, groups, 0, 3);
        kernel.insert_before(end, Op::Barrier);
    }

    // ---- the mainloop: groups are a chain, the fragment pairs are emitted
    // one after the other because their `ldmatrix` words are registers ----
    // The instruction broadcasts a one-component accumulator into both of its
    // C words, so the zero is a bare scalar, not a packed fragment.
    let zero = kernel.insert_before(end, Op::Const(Constant::new(0i32)));
    let n_groups = usize::try_from(groups).expect("K groups per chunk");
    let n_sub = usize::try_from(mt_w / 2).expect("fragment pairs per warp");
    let n_bands = usize::try_from(nt_w).expect("activation bands per warp");
    for g in 0..n_groups {
        let gv = kernel.insert_before(end, Op::Const(Constant::idx(i64::try_from(g).expect("K group"))));
        let g8 = {
            // One group is `Q8_1_GROUP` elements, `Q8_1_GROUP / 4` words.
            let gw = kernel.insert_before(end, Op::Const(Constant::idx(Q8_1_GROUP / 4)));
            kernel.insert_before(end, Op::Binary { x: gv, y: gw, bop: BOp::Mul })
        };
        for sub in 0..n_sub {
            let s16 = kernel.insert_before(end, Op::Const(Constant::idx(i64::try_from(sub * 2).expect("sub-tile") * fm)));
            // The two A fragments' words, indexed `fragment + 2 * K step`: a
            // staged tile hands all four back from one `ldmatrix.x4`, an
            // unstaged form reads each fragment word straight from the cone.
            let (aw0, aw1, aw2, aw3) = if staged {
                let mut row = add_off(kernel, s16, warp_off_m);
                row = kernel.insert_before(end, Op::Binary { x: row, y: lanerow, bop: BOp::Add });
                let mut addr = kernel.insert_before(end, Op::Binary { x: row, y: cxs, bop: BOp::Mul });
                addr = kernel.insert_before(end, Op::Binary { x: addr, y: g8, bop: BOp::Add });
                let side = {
                    let four = kernel.insert_before(end, Op::Const(Constant::idx(4i64)));
                    kernel.insert_before(end, Op::Binary { x: lanehalf, y: four, bop: BOp::Mul })
                };
                addr = kernel.insert_before(end, Op::Binary { x: addr, y: side, bop: BOp::Add });
                let frag = kernel.insert_before(
                    end,
                    Op::PTX(Box::new(PTXOp::VLdMatrix {
                        dims: LdMatrixDims::m8n8,
                        num: LdMatrixNum::X4,
                        trans: LdMatrixTrans::off,
                        dtype: LdMatrixType::b16,
                        src: xt,
                        addr,
                    })),
                );
                let aw0 = kernel.insert_before(end, Op::Index { vec: frag, idx: 0 });
                let aw1 = kernel.insert_before(end, Op::Index { vec: frag, idx: 1 });
                let aw2 = kernel.insert_before(end, Op::Index { vec: frag, idx: 2 });
                let aw3 = kernel.insert_before(end, Op::Index { vec: frag, idx: 3 });
                (aw0, aw1, aw2, aw3)
            } else {
                let mut words = [OpId::NULL; 4];
                for f in 0..2usize {
                    // The fragment's row: this lane holds row `lane / 4` of it.
                    for h in 0..2usize {
                        let mut row = add_off(kernel, s16, warp_off_m);
                        let foff = kernel.insert_const_idx_before(end, Dim::from(f as i64) * fm);
                        row = kernel.insert_before(end, Op::Binary { x: row, y: foff, bop: BOp::Add });
                        row = kernel.insert_before(end, Op::Binary { x: row, y: gid, bop: BOp::Add });
                        let row = kernel.insert_before(end, Op::Binary { x: row, y: a_roff, bop: BOp::Add });
                        // The word's K origin: the group's own start, the K
                        // step, and this lane's four codes within it.
                        let kg = kernel.insert_before(
                            end,
                            Op::Const(Constant::idx(
                                Dim::from(i64::try_from(g).expect("group")) * Q8_1_GROUP + Dim::from(h as i64) * fk,
                            )),
                        );
                        let mut k = kernel.insert_before(end, Op::Binary { x: kg, y: a_koff, bop: BOp::Add });
                        let t4 = kernel.insert_before(end, Op::Const(Constant::idx(4i64)));
                        let tigk = kernel.insert_before(end, Op::Binary { x: tig, y: t4, bop: BOp::Mul });
                        k = kernel.insert_before(end, Op::Binary { x: k, y: tigk, bop: BOp::Add });
                        words[f + 2 * h] = cone_word4(kernel, end, qa, row, k);
                    }
                }
                (words[0], words[1], words[2], words[3])
            };
            // The two A fragments' group scales.
            let (da0, da1) = if staged {
                let mut crow = add_off(kernel, s16, warp_off_m);
                crow = kernel.insert_before(end, Op::Binary { x: crow, y: gid, bop: BOp::Add });
                let mut i0 = kernel.insert_before(end, Op::Binary { x: crow, y: cxs, bop: BOp::Mul });
                i0 = kernel.insert_before(end, Op::Binary { x: i0, y: cpay, bop: BOp::Add });
                i0 = kernel.insert_before(end, Op::Binary { x: i0, y: gv, bop: BOp::Add });
                let w0 = kernel.insert_before(end, Op::Load { src: xt, src_idx: i0 });
                let d0 = kernel.insert_before(end, Op::Bitcast { x: w0, dtype: DType::F32 });
                let eight = kernel.insert_before(end, Op::Const(Constant::idx(fm)));
                let mut i1 = kernel.insert_before(end, Op::Binary { x: crow, y: eight, bop: BOp::Add });
                i1 = kernel.insert_before(end, Op::Binary { x: i1, y: cxs, bop: BOp::Mul });
                i1 = kernel.insert_before(end, Op::Binary { x: i1, y: cpay, bop: BOp::Add });
                i1 = kernel.insert_before(end, Op::Binary { x: i1, y: gv, bop: BOp::Add });
                let w1 = kernel.insert_before(end, Op::Load { src: xt, src_idx: i1 });
                let d1 = kernel.insert_before(end, Op::Bitcast { x: w1, dtype: DType::F32 });
                (d0, d1)
            } else {
                let kg =
                    kernel.insert_before(end, Op::Const(Constant::idx(Dim::from(i64::try_from(g).expect("group")) * Q8_1_GROUP)));
                let k = kernel.insert_before(end, Op::Binary { x: kg, y: a_koff, bop: BOp::Add });
                let mut s = [OpId::NULL; 2];
                for (f, slot) in s.iter_mut().enumerate() {
                    let mut row = add_off(kernel, s16, warp_off_m);
                    let foff = kernel.insert_const_idx_before(end, Dim::from(f as i64) * fm);
                    row = kernel.insert_before(end, Op::Binary { x: row, y: foff, bop: BOp::Add });
                    row = kernel.insert_before(end, Op::Binary { x: row, y: gid, bop: BOp::Add });
                    let row = kernel.insert_before(end, Op::Binary { x: row, y: a_roff, bop: BOp::Add });
                    let v = qa.scale.clone_at_coords(kernel, end, row, k);
                    *slot = if kernel.dtype(v) == DType::F32 {
                        v
                    } else {
                        kernel.insert_before(end, Op::Cast { x: v, dtype: DType::F32 })
                    };
                }
                (s[0], s[1])
            };
            for bj in 0..n_bands {
                let bj8 = kernel.insert_before(end, Op::Const(Constant::idx(i64::try_from(bj).expect("activation band") * fn_)));
                let (b0, b1) = if staged {
                    let mut brow = add_off(kernel, bj8, warp_off_n);
                    brow = kernel.insert_before(end, Op::Binary { x: brow, y: gid, bop: BOp::Add });
                    let brow_base = kernel.insert_before(end, Op::Binary { x: brow, y: cys, bop: BOp::Mul });
                    let mut bcol = kernel.insert_before(end, Op::Binary { x: brow_base, y: cgrp, bop: BOp::Add });
                    bcol = kernel.insert_before(end, Op::Binary { x: bcol, y: g8, bop: BOp::Add });
                    bcol = kernel.insert_before(end, Op::Binary { x: bcol, y: tig, bop: BOp::Add });
                    let b0 = kernel.insert_before(end, Op::Load { src: yt, src_idx: bcol });
                    let b1 = {
                        // The second half of the K step: `fk / 4` words on.
                        let half = kernel.insert_before(end, Op::Const(Constant::idx(fk / 4)));
                        let i = kernel.insert_before(end, Op::Binary { x: bcol, y: half, bop: BOp::Add });
                        kernel.insert_before(end, Op::Load { src: yt, src_idx: i })
                    };
                    (b0, b1)
                } else {
                    let mut w = [OpId::NULL; 2];
                    for (h, slot) in w.iter_mut().enumerate() {
                        let mut row = add_off(kernel, bj8, warp_off_n);
                        row = kernel.insert_before(end, Op::Binary { x: row, y: gid, bop: BOp::Add });
                        let row = kernel.insert_before(end, Op::Binary { x: row, y: b_roff, bop: BOp::Add });
                        let kg = kernel.insert_before(
                            end,
                            Op::Const(Constant::idx(
                                Dim::from(i64::try_from(g).expect("group")) * Q8_1_GROUP + Dim::from(h as i64) * fk,
                            )),
                        );
                        let mut k = kernel.insert_before(end, Op::Binary { x: kg, y: b_koff, bop: BOp::Add });
                        let t4 = kernel.insert_before(end, Op::Const(Constant::idx(4i64)));
                        let tigk = kernel.insert_before(end, Op::Binary { x: tig, y: t4, bop: BOp::Mul });
                        k = kernel.insert_before(end, Op::Binary { x: k, y: tigk, bop: BOp::Add });
                        *slot = cone_word4(kernel, end, qb, row, k);
                    }
                    (w[0], w[1])
                };
                // The two C columns' activation scales: rows `2*tig + word`.
                let (db0, db1) = if staged {
                    let mut r0 = add_off(kernel, bj8, warp_off_n);
                    r0 = kernel.insert_before(end, Op::Binary { x: r0, y: tig2, bop: BOp::Add });
                    let i0 = kernel.insert_before(end, Op::Binary { x: r0, y: cys, bop: BOp::Mul });
                    let i0 = kernel.insert_before(end, Op::Binary { x: i0, y: gv, bop: BOp::Add });
                    let w0 = kernel.insert_before(end, Op::Load { src: yt, src_idx: i0 });
                    let d0 = kernel.insert_before(end, Op::Bitcast { x: w0, dtype: DType::F32 });
                    let one = kernel.insert_before(end, Op::Const(Constant::idx(1i64)));
                    let r1 = kernel.insert_before(end, Op::Binary { x: r0, y: one, bop: BOp::Add });
                    let i1 = kernel.insert_before(end, Op::Binary { x: r1, y: cys, bop: BOp::Mul });
                    let i1 = kernel.insert_before(end, Op::Binary { x: i1, y: gv, bop: BOp::Add });
                    let w1 = kernel.insert_before(end, Op::Load { src: yt, src_idx: i1 });
                    let d1 = kernel.insert_before(end, Op::Bitcast { x: w1, dtype: DType::F32 });
                    (d0, d1)
                } else {
                    let kg = kernel
                        .insert_before(end, Op::Const(Constant::idx(Dim::from(i64::try_from(g).expect("group")) * Q8_1_GROUP)));
                    let k = kernel.insert_before(end, Op::Binary { x: kg, y: b_koff, bop: BOp::Add });
                    let mut s = [OpId::NULL; 2];
                    for (slot, s) in s.iter_mut().enumerate() {
                        let mut r0 = add_off(kernel, bj8, warp_off_n);
                        r0 = kernel.insert_before(end, Op::Binary { x: r0, y: tig2, bop: BOp::Add });
                        if slot == 1 {
                            let one = kernel.insert_before(end, Op::Const(Constant::idx(1i64)));
                            r0 = kernel.insert_before(end, Op::Binary { x: r0, y: one, bop: BOp::Add });
                        }
                        let row = kernel.insert_before(end, Op::Binary { x: r0, y: b_roff, bop: BOp::Add });
                        let v = qb.scale.clone_at_coords(kernel, end, row, k);
                        *s = if kernel.dtype(v) == DType::F32 {
                            v
                        } else {
                            kernel.insert_before(end, Op::Cast { x: v, dtype: DType::F32 })
                        };
                    }
                    (s[0], s[1])
                };
                for par in 0..2usize {
                    let (aw, aw_next) = if par == 0 { (aw0, aw2) } else { (aw1, aw3) };
                    let c0 = kernel
                        .insert_before(end, Op::Wmma { dims: v.dims, layout: v.layout, dtype: v.dtype, a: aw, b: b0, c: zero });
                    let c1 = kernel.insert_before(
                        end,
                        Op::Wmma { dims: v.dims, layout: v.layout, dtype: v.dtype, a: aw_next, b: b1, c: c0 },
                    );
                    let ai = sub * 2 + par;
                    let base = (ai * n_bands + bj) * usize::try_from(slots).expect("C words");
                    for slot in 0..usize::try_from(slots).expect("C words") {
                        let iv = kernel.insert_before(end, Op::Index { vec: c1, idx: slot });
                        let fv = kernel.insert_before(end, Op::Cast { x: iv, dtype: DType::F32 });
                        let da = if par == 0 { da0 } else { da1 };
                        let db = if slot == 0 { db0 } else { db1 };
                        let sc = kernel.insert_before(end, Op::Binary { x: da, y: db, bop: BOp::Mul });
                        let prod = kernel.insert_before(end, Op::Binary { x: fv, y: sc, bop: BOp::Mul });
                        let idx =
                            kernel.insert_before(end, Op::Const(Constant::idx(i64::try_from(base + slot).expect("sum index"))));
                        let prev = kernel.insert_before(end, Op::Load { src: t.acc, src_idx: idx });
                        let sum = kernel.insert_before(end, Op::Binary { x: prod, y: prev, bop: BOp::Add });
                        kernel.insert_before(end, Op::Store { dst: t.acc, dst_idx: idx, src: sum });
                    }
                }
            }
        }
    }
    kernel.insert_before(end, Op::Barrier);

    // ---- the write-back: the same C fragment geometry as the other forms ----
    for ai in 0..usize::try_from(mt_w).expect("m fragments") {
        for bj in 0..usize::try_from(nt_w).expect("n fragments") {
            let combo = put(kernel, Op::Const(Constant::idx(i64::try_from(ai * n_bands + bj).expect("fragment lane"))));
            let sl = put(kernel, Op::Unroll { axis: 2, len: u16::try_from(slots).expect("C words") });
            scatter_epilogue(kernel, t, combo, sl, gid, tig, nt_w, slots, fm, fn_, warp_off_m, warp_off_n);
        }
    }
    let mut doomed: Vec<OpId> = vec![t.update];
    doomed.extend(t.epilogue.iter().map(|e| e.store));
    for id in doomed {
        kernel.remove_op(id);
    }
    kernel.verify();
}

/// Scatter the region's f32 accumulator to its output stores: each C word of
/// the instruction is written at its own coordinate — `row` is the fragment's
/// row (the warp's band included), the column the fragment's column
/// `2*(lane%4) + word` plus the warp's band and the tile's column — over the
/// warp's tile lane.
#[allow(clippy::too_many_arguments)]
fn scatter_epilogue(
    kernel: &mut Kernel,
    t: &Tiled,
    combo: OpId,
    sl: OpId,
    gid: OpId,
    tig: OpId,
    nt_w: Dim,
    slots: Dim,
    fm: Dim,
    fn_: Dim,
    warp_off_m: OpId,
    warp_off_n: OpId,
) {
    for epi in &t.epilogue {
        // The lane's fragment geometry, rebuilt here: the lane nodes are the
        // block's, everything derived from them is local to its use site.
        let ntc = kernel.insert_const_idx_before(epi.store, nt_w);
        let ai = kernel.insert_before(epi.store, Op::Binary { x: combo, y: ntc, bop: BOp::Div });
        let bj = kernel.insert_before(epi.store, Op::Binary { x: combo, y: ntc, bop: BOp::Mod });
        let slots_c = kernel.insert_const_idx_before(epi.store, slots);
        let idx = kernel.insert_before(epi.store, Op::Binary { x: combo, y: slots_c, bop: BOp::Mul });
        let idx = kernel.insert_before(epi.store, Op::Binary { x: idx, y: sl, bop: BOp::Add });
        let two = kernel.insert_const_idx_before(epi.store, Dim::from(2));
        let tig2 = kernel.insert_before(epi.store, Op::Binary { x: tig, y: two, bop: BOp::Mul });
        let fmc = kernel.insert_const_idx_before(epi.store, fm);
        let mut row = kernel.insert_before(epi.store, Op::Binary { x: ai, y: fmc, bop: BOp::Mul });
        row = kernel.insert_before(epi.store, Op::Binary { x: row, y: gid, bop: BOp::Add });
        if !warp_off_m.is_null() {
            row = kernel.insert_before(epi.store, Op::Binary { x: row, y: warp_off_m, bop: BOp::Add });
        }
        let f = kernel.insert_const_idx_before(epi.store, fn_);
        let mut col = kernel.insert_before(epi.store, Op::Binary { x: bj, y: f, bop: BOp::Mul });
        col = kernel.insert_before(epi.store, Op::Binary { x: col, y: tig2, bop: BOp::Add });
        col = kernel.insert_before(epi.store, Op::Binary { x: col, y: sl, bop: BOp::Add });
        if !warp_off_n.is_null() {
            col = kernel.insert_before(epi.store, Op::Binary { x: col, y: warp_off_n, bop: BOp::Add });
        }
        let mut elem = kernel.insert_before(epi.store, Op::Load { src: t.acc, src_idx: idx });
        for dtype in epi.chain.iter().rev() {
            elem = kernel.insert_before(epi.store, Op::Cast { x: elem, dtype: *dtype });
        }
        let mut addr = base_expr(kernel, epi.store, &epi.fixed, epi.offset, 0);
        let rowt = if epi.m != 1 {
            let st = kernel.insert_const_idx_before(epi.store, epi.m);
            kernel.insert_before(epi.store, Op::Binary { x: row, y: st, bop: BOp::Mul })
        } else {
            row
        };
        addr = kernel.insert_before(epi.store, Op::Binary { x: addr, y: rowt, bop: BOp::Add });
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

/// Emit a matched tiled form: the int8 instruction gets its own emission (its
/// operand words and per-group scales are staged as bytes and rescaled once
/// per group), every f16 instruction the shared one.
fn tile_to_wmma_form(kernel: &mut Kernel, t: &Tiled, staged: bool, swizzle: bool, layout: (u16, u16, u16)) {
    if matches!(t.variant.dtype, MMADType::s8_s8_s32_s32) {
        tile_to_wmma_s8_cell(kernel, t, staged, swizzle, layout);
    } else {
        tile_to_wmma(kernel, t, staged, swizzle, layout);
    }
}

/// Emit a matched tiled region as wmma instructions, in place: the
/// accumulator storage keeps its identity, its accesses are rewritten in
/// fragment order, and the split's reduce/update and the output stores are
/// replaced (DCE drops their cones).
fn tile_to_wmma(kernel: &mut Kernel, t: &Tiled, staged: bool, swizzle: bool, layout: (u16, u16, u16)) {
    let v = &t.variant;
    let (fm, fn_, fk) = {
        let (m, n, k) = v.dims.decompose_mnk();
        (m as Dim, n as Dim, k as Dim)
    };
    let info = kernel.device_info();
    let warp = u32::from(info.warp_size);
    let Op::Loop { len: loop_len_id } = *kernel.at(t.loop_id) else {
        unreachable!("tiled tensorize: the K loop is not a loop")
    };
    let shape = TileShape { mt: t.mt, nt: t.nt, kt: t.kt, fm, fn_: fn_, fk, a_k: t.a_k, b_k: t.b_k, slots: t.slots };
    let Some(fit) = tile_fit(&shape, staged, layout, &info, |wk| kernel.is_sym_multiple(loop_len_id, Dim::from(wk))) else {
        panic!(
            "tile_to_wmma: no form for tile {:?}x{:?}x{:?} (instruction {fm:?}x{fn_:?}x{fk:?}, K step {:?}x{:?}) at layout {layout:?} staged={staged}: the enumeration must not offer it",
            t.mt, t.nt, t.kt, t.a_k, t.b_k
        );
    };
    let (wm, wn, wk) = layout;
    let TileFit { mt_w, nt_w, warps, combine, staged, chunk, interleave } = fit;
    let mut warp_var = OpId::NULL;
    let mut warp_off_a = OpId::NULL;
    let mut warp_off_b = OpId::NULL;
    let mut warp_off_m = OpId::NULL;
    let mut warp_off_n = OpId::NULL;
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
    // The threads one block-wide fill covers: every thread of the block.
    let threads = Dim::from(warp);

    // The accumulator holds one register per (tile, slot): `m_len * n_len`
    // scalar elements would be 32x what the instruction layout needs.
    // A warp grid hands each warp its own band of the tile grid, so the
    // accumulator only holds that band: the whole grid's `mt * nt * slots`
    // would be eight times the registers a warp uses.
    let len = mt_w * nt_w * t.slots;
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
        // `ldmatrix` returns the instruction's destination list: one 32-bit
        // word per 8x8 matrix, so an operand of `slots` elements is
        // `slots / elements-per-word` words of the load. The operand dtype
        // owns the word width.
        let a_per_word = usize::from(32 / v.a_dtype.bit_size());
        let b_per_word = usize::from(32 / v.b_dtype.bit_size());
        let a_words = a_slots / a_per_word;
        let b_words = b_slots / b_per_word;
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
                let (asl, av) = match &t.a_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, t.update, fill_lane, cone, a_base, t.a_m, t.a_k, fm, chunk, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        t.update,
                        fill_lane,
                        t.a_src,
                        a_base,
                        t.a_m,
                        t.a_k,
                        fm,
                        chunk,
                        fill_threads,
                        swizzle,
                    ),
                };
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
                            chunk / LDMATRIX_EDGE,
                            sb,
                            OpId::NULL,
                            swizzle,
                        );
                        a_frags[tm][tk] = slice(kernel, t.update, value, false, a_words);
                        a_frags[tm][tk + 1] = slice(kernel, t.update, value, true, a_words);
                        tk += 2;
                    } else {
                        a_frags[tm][tk] = coop_ldmatrix(
                            kernel,
                            t.update,
                            lane,
                            a_tile,
                            fm,
                            fk,
                            fm,
                            tk as Dim * fk,
                            chunk / LDMATRIX_EDGE,
                            sb,
                            OpId::NULL,
                            swizzle,
                        );
                        tk += 1;
                    }
                }
            }
            for tn in 0..nt {
                let b_base = base_expr(kernel, t.update, &b_fixed, t.b_offset, tn as Dim * fn_ * t.b_n);
                let b_base = shift(kernel, t.update, b_base, [warp_off_b, b_band]);
                let sb = b_slot;
                sync(kernel);
                let (bsl, bv) = match &t.b_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, t.update, fill_lane, cone, b_base, t.b_n, t.b_k, fn_, chunk, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        t.update,
                        fill_lane,
                        t.b_src,
                        b_base,
                        t.b_n,
                        t.b_k,
                        fn_,
                        chunk,
                        fill_threads,
                        swizzle,
                    ),
                };
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
                            chunk / LDMATRIX_EDGE,
                            sb,
                            OpId::NULL,
                            swizzle,
                        );
                        for (step, frag) in [
                            (tk, slice(kernel, t.update, value, false, b_words)),
                            (tk + 1, slice(kernel, t.update, value, true, b_words)),
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
                            chunk / LDMATRIX_EDGE,
                            sb,
                            OpId::NULL,
                            swizzle,
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
                let (asl, av) = match &t.a_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, t.loop_id, fill_lane, cone, a0, t.a_m, t.a_k, a_rows, win_k, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        t.loop_id,
                        fill_lane,
                        t.a_src,
                        a0,
                        t.a_m,
                        t.a_k,
                        a_rows,
                        win_k,
                        fill_threads,
                        swizzle,
                    ),
                };
                coop_fill_store(kernel, t.loop_id, ab_buf, OpId::NULL, asl, av);
                let (bsl, bv) = match &t.b_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, t.loop_id, fill_lane, cone, b0, t.b_n, t.b_k, b_rows, win_k, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        t.loop_id,
                        fill_lane,
                        t.b_src,
                        b0,
                        t.b_n,
                        t.b_k,
                        b_rows,
                        win_k,
                        fill_threads,
                        swizzle,
                    ),
                };
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
            //
            // A pair then holds `steps_here` K steps: one per step the window
            // staged, divided over its pairs. A one-step window (`kt < 2`) has
            // a single unpaired `ldmatrix` per pair, so only `sl = 0` exists —
            // reading a second step addresses one tile run past the section.
            let npairs = (win_steps / 2).max(1);
            let steps_here = win_steps / npairs;
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
                let (asl, av) = match &t.a_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, end, fill_lane, cone, a_fill, t.a_m, t.a_k, a_rows, win_k, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        end,
                        fill_lane,
                        t.a_src,
                        a_fill,
                        t.a_m,
                        t.a_k,
                        a_rows,
                        win_k,
                        fill_threads,
                        swizzle,
                    ),
                };
                let (bsl, bv) = match &t.b_cone {
                    Some(cone) => {
                        coop_fill_cone(kernel, end, fill_lane, cone, b_fill, t.b_n, t.b_k, b_rows, win_k, fill_threads, swizzle)
                    }
                    None => coop_fill_load(
                        kernel,
                        end,
                        fill_lane,
                        t.b_src,
                        b_fill,
                        t.b_n,
                        t.b_k,
                        b_rows,
                        win_k,
                        fill_threads,
                        swizzle,
                    ),
                };
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
                // `width` counts the fragment's elements; the `ldmatrix` result
                // and the staged section are 32-bit words, one word per
                // `elements-per-word` of the operand dtype.
                let a_per_word = usize::from(32 / v.a_dtype.bit_size());
                let b_per_word = usize::from(32 / v.b_dtype.bit_size());
                let wwidth = width / a_per_word;
                let a_w = a_slots / a_per_word;
                let b_w = b_slots / b_per_word;
                let nlanes = 2 * npairs;
                let ab_st =
                    insert_scoped_storage(kernel, DType::U32, MemScope::Register, Dim::from((nlanes * mt * wwidth) as i64));
                let obl =
                    kernel.insert_before(end, Op::Unroll { axis: 3, len: u16::try_from(nlanes).expect("operand-pair lane") });
                let tlane = kernel.insert_before(end, Op::Unroll { axis: 0, len: u16::try_from(mt).expect("tile count") });
                // The lane's slot: its pair section, its tile, and the warp's own
                // band (A's rows for A's lanes, B's for B's). A section is the
                // K steps its pair covers — two per `ldmatrix`, or the window's
                // single step when it has no pair to share.
                let pair_k = i64::try_from(steps_here).expect("K steps per pair") * fk;
                let ps = kernel.insert_const_idx_before(end, pair_k * a_rows);
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
                        win_k / LDMATRIX_EDGE,
                        sb,
                        a_lb,
                        swizzle,
                    );
                    // `[lane][tile][half]`: one contiguous run per tile, so one
                    // store per lane writes the whole `ldmatrix` result.
                    let sect = kernel.insert_const_idx_before(end, Dim::from((mt * wwidth) as i64));
                    let base = kernel.insert_before(end, Op::Binary { x: obl, y: sect, bop: BOp::Mul });
                    let ws = kernel.insert_const_idx_before(end, Dim::from(wwidth as i64));
                    let tb = kernel.insert_before(end, Op::Binary { x: tlane, y: ws, bop: BOp::Mul });
                    let base = kernel.insert_before(end, Op::Binary { x: base, y: tb, bop: BOp::Add });
                    let wl =
                        kernel.insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(wwidth).expect("fragment width") });
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
                        let asec = kernel.insert_const_idx_before(end, Dim::from((pi * mt * wwidth) as i64));
                        let ws = kernel.insert_const_idx_before(end, Dim::from(wwidth as i64));
                        let abase = kernel.insert_before(end, Op::Binary { x: tml, y: ws, bop: BOp::Mul });
                        let abase = kernel.insert_before(end, Op::Binary { x: abase, y: asec, bop: BOp::Add });
                        let am = kernel.insert_const_idx_before(end, Dim::from((sl * a_w) as i64));
                        let abase = kernel.insert_before(end, Op::Binary { x: abase, y: am, bop: BOp::Add });
                        let alane =
                            kernel.insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(a_w).expect("a fragment slots") });
                        let aidx = kernel.insert_before(end, Op::Binary { x: abase, y: alane, bop: BOp::Add });
                        let a_frag = kernel.insert_before(end, Op::Load { src: ab_st, src_idx: aidx });
                        // B: operand 1, the `ldmatrix`'s own word order:
                        // `(tile-in-pair) + 2 * K step`, one `b_slots` each.
                        let bsec = kernel.insert_const_idx_before(end, Dim::from(((npairs + pi) * mt * wwidth) as i64));
                        let two = kernel.insert_const_idx_before(end, Dim::from(2i64));
                        let tpair = kernel.insert_before(end, Op::Binary { x: tnl, y: two, bop: BOp::Div });
                        let bbase = kernel.insert_before(end, Op::Binary { x: tpair, y: ws, bop: BOp::Mul });
                        let bbase = kernel.insert_before(end, Op::Binary { x: bbase, y: bsec, bop: BOp::Add });
                        let trem = kernel.insert_before(end, Op::Binary { x: tnl, y: two, bop: BOp::Mod });
                        let tmn = kernel.insert_const_idx_before(end, Dim::from(2 * sl as i64));
                        let tidx = kernel.insert_before(end, Op::Binary { x: trem, y: tmn, bop: BOp::Add });
                        let bsl = kernel.insert_const_idx_before(end, Dim::from(b_w as i64));
                        let bmid = kernel.insert_before(end, Op::Binary { x: tidx, y: bsl, bop: BOp::Mul });
                        let bbase = kernel.insert_before(end, Op::Binary { x: bbase, y: bmid, bop: BOp::Add });
                        let blane =
                            kernel.insert_before(end, Op::Unroll { axis: 1, len: u16::try_from(b_w).expect("b fragment slots") });
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
                    let a_elems =
                        frag_elems(kernel, t.update, lane, &v.a, t.a_src, t.a_cone.as_ref(), a_base, t.a_m, t.a_k, warp);
                    let a_frag = kernel.insert_before(t.update, Op::Stack { ops: a_elems.into() });
                    let b_elems =
                        frag_elems(kernel, t.update, lane, &v.b, t.b_src, t.b_cone.as_ref(), b_base, t.b_n, t.b_k, warp);
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
            // Each side is a plain load or a fused dequant chain over the same
            // affine element index; both name the operand's logical view.
            let (Some((a_src, a_cone, a_idx)), Some((b_src, b_cone, b_idx))) =
                (operand_view(self, la, false), operand_view(self, lb, false))
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
            let (a_map, b_map, a_src, b_src, a_cone, b_cone, a_val, b_val) =
                if map_la.contains_key(&m_range) && map_lb.contains_key(&n_range) {
                    (map_la, map_lb, a_src, b_src, a_cone, b_cone, la, lb)
                } else if map_la.contains_key(&n_range) && map_lb.contains_key(&m_range) {
                    (map_lb, map_la, b_src, a_src, b_cone, a_cone, lb, la)
                } else {
                    continue;
                };

            // Operand and accumulator dtypes, as the views are oriented. A
            // quantized operand's dtype is its chain's value (the dequantized
            // f16 or the int8 code), never the packed storage's.
            let a_dtype = self.dtype(a_val);
            let b_dtype = self.dtype(b_val);
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
            //
            // Both forms of a quantized operand are enumerated, and the search
            // decides between them: the f16 form reads the dequantized value
            // (the chain evaluated per element into an f16 tile), the int8 form
            // reads the code and applies the group scale to the int32
            // accumulator. The f16 form needs the operands' *values* to be f16;
            // the int8 form needs both chains to expose a code and a per-group
            // scale, and keeps the region's f32 accumulator as the scaled sum.
            for variant in &info.mma_variants {
                let s8 = matches!(variant.dtype, MMADType::s8_s8_s32_s32);
                if s8 {
                    // The int8 form is emitted by the tiled (staged) form only:
                    // its codes and group scales live in a staged tile, so there
                    // is no single-warp region form for it.
                    continue;
                }
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
                    a_cone: a_cone.clone(),
                    a_m: a_map[&m_range].1,
                    a_k: a_map[&k_loop].1,
                    b_src,
                    b_cone: b_cone.clone(),
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
            if std::env::var("ZYX_TD").is_ok() {
                eprintln!("TD tiled block entered");
            }
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
                let (Some((a_src, a_cone, a_idx)), Some((b_src, b_cone, b_idx))) =
                    (operand_view(self, la, true), operand_view(self, lb, true))
                else {
                    if std::env::var("ZYX_TD").is_ok() {
                        let ka = Cone::at(self, la);
                        let kb = Cone::at(self, lb);
                        eprintln!(
                            "TD operand_view failed: la={:?} lb={:?} ca={:?} cb={:?}",
                            self.at(la),
                            self.at(lb),
                            ka.map(|c| c.ops.len()),
                            kb.map(|c| c.ops.len())
                        );
                    }
                    continue;
                };
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

                if std::env::var("ZYX_TD").is_ok() {
                    eprintln!(
                        "TD reached: a_map={:?} b_map={:?} m_var={:?} n_var={:?} k_var={:?} epi={}",
                        a_map.keys().filter(|v| !v.is_null()).count(),
                        b_map.keys().filter(|v| !v.is_null()).count(),
                        m_var,
                        n_var,
                        k_var,
                        epilogue.len()
                    );
                }
                let a_dtype = self.dtype(la);
                let b_dtype = self.dtype(lb);
                let c_dtype = match self.at(acc) {
                    Op::Storage { dtype, .. } => *dtype,
                    _ => continue,
                };
                for variant in &info.mma_variants {
                    let s8 = matches!(variant.dtype, MMADType::s8_s8_s32_s32);
                    if s8 {
                        // The int8 form consumes the codes and rescales their
                        // int32 sums into the region's f32 accumulator; it is
                        // emitted only staged (the staged tile is where the
                        // codes and the group scales are laid out).
                        if b_cone.is_none() || c_dtype != DType::F32 {
                            continue;
                        }
                    } else if variant.a_dtype != a_dtype || variant.b_dtype != b_dtype || variant.c_dtype != c_dtype {
                        continue;
                    }
                    let (fm, fn_, fk) = {
                        let (fm, fn_, fk) = variant.dims.decompose_mnk();
                        (fm as Dim, fn_ as Dim, fk as Dim)
                    };
                    if k_len % fk != 0 {
                        continue;
                    }
                    // A wider K step unrolls the loop body instead of adding
                    // iterations, so the ceiling is the emitted body: the
                    // staged form's is one loop over its fragments whatever
                    // the tile grid, the unstage form's is the grid itself
                    // (bounded where that body is emitted).
                    let kt = k_len / fk;
                    // The direct tiling: A is the matched product's first
                    // operand, so its lanes must be whole `fm` tiles.
                    if m_len % fm == 0 && n_len % fn_ == 0 {
                        let (mt, nt) = (m_len / fm, n_len / fn_);
                        if mt.saturating_mul(nt).saturating_mul(kt) <= 2048 {
                            tiled.push(Tiled {
                                variant: variant.clone(),
                                acc,
                                a_src,
                                b_src,
                                a_cone: a_cone.clone(),
                                b_cone: b_cone.clone(),
                                a_quant: a_cone.as_ref().and_then(|c| Quant::at(self, c)),
                                b_quant: b_cone.as_ref().and_then(|c| Quant::at(self, c)),
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
                                mirrored: false,
                            });
                        }
                    }
                    // The same reduce with the operand roles exchanged: A gets
                    // the second operand's tile and B the first, and the m/n
                    // tile extents and output strides swap with them. The
                    // extents are recomputed from the exchanged lane lengths
                    // (A's tile is `fm` wide, B's `fn_`). Both orientations are
                    // valid kernels (the output index is addressed by the
                    // lanes' own strides), and the search enumerates both, so
                    // the form is emitted for each. Equal operand dtypes are
                    // what the instruction requires of either role, and each
                    // side must be a whole number of instructions.
                    if variant.a_dtype == variant.b_dtype && n_len % fm == 0 && m_len % fn_ == 0 {
                        let (mt, nt) = (n_len / fm, m_len / fn_);
                        if mt.saturating_mul(nt).saturating_mul(kt) <= 2048 {
                            tiled.push(Tiled {
                                variant: variant.clone(),
                                acc,
                                a_src: b_src,
                                b_src: a_src,
                                a_cone: b_cone.clone(),
                                b_cone: a_cone.clone(),
                                a_quant: b_cone.as_ref().and_then(|c| Quant::at(self, c)),
                                b_quant: a_cone.as_ref().and_then(|c| Quant::at(self, c)),
                                a_fixed: b_fixed.clone(),
                                a_offset: b_offset,
                                a_m: b_map[&n_var].1,
                                a_k: b_map[&k_var].1,
                                b_fixed: a_fixed.clone(),
                                b_offset: a_offset,
                                b_n: a_map[&m_var].1,
                                b_k: a_map[&k_var].1,
                                mt,
                                nt,
                                kt,
                                slots: Dim::from(variant.c.slots.len() as i64),
                                init,
                                loop_id,
                                update,
                                epilogue: epilogue.iter().map(|e| Epi { m: e.n, n: e.m, ..e.clone() }).collect(),
                                mirrored: true,
                            });
                        }
                    }
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
            // A pre-split form is offered only at the warp layouts and staging
            // levels that can emit it: the form is applied as-is, so a request
            // it cannot honour is a panic, not a config.
            let shape = tiled_idx.map(|idx| {
                let t = &tiled[idx];
                let (fm, fn_, fk) = {
                    let (m, n, k) = t.variant.dims.decompose_mnk();
                    (m as Dim, n as Dim, k as Dim)
                };
                let Op::Loop { len: loop_len } = *self.at(t.loop_id) else {
                    unreachable!("tiled tensorize: the K loop is not a loop")
                };
                (TileShape { mt: t.mt, nt: t.nt, kt: t.kt, fm, fn_: fn_, fk, a_k: t.a_k, b_k: t.b_k, slots: t.slots }, loop_len)
            });
            let coned = region
                .map(|r| regions[r].a_cone.is_some() || regions[r].b_cone.is_some())
                .or_else(|| tiled_idx.map(|t| tiled[t].a_cone.is_some() || tiled[t].b_cone.is_some()))
                .unwrap_or(false);
            let s8_form = region
                .map(|r| matches!(regions[r].variant.dtype, MMADType::s8_s8_s32_s32))
                .or_else(|| tiled_idx.map(|t| matches!(tiled[t].variant.dtype, MMADType::s8_s8_s32_s32)))
                .unwrap_or(false);
            // A fused-dequant operand's value is a chain over packed words, not
            // an element stream: the element form's fill re-derives that chain
            // per f16 element into a tile whose addressing is not the chain's,
            // which the device rejects (an illegal address at launch) or a
            // `verify` dtype mismatch refuses. Only the int8 cell — which stages
            // the chain's codes and per-group scales at the format's own word
            // width — is a candidate for a coned operand.
            if coned && !s8_form {
                continue;
            }
            for &layout in &layouts {
                for staged in [false, true] {
                    if !staged && layout.2 > 1 {
                        continue;
                    }
                    // A dequant chain is staged: reading it per fragment
                    // register re-evaluates the whole chain, and the device
                    // epilogue then unfolds each chain per fragment lane, an IR
                    // blowout that never runs faster than the staged tile. The
                    // int8 cell is the exception: its fragment operand *is* one
                    // code word, so an unstaged form reads that word once per
                    // instruction instead of unfolding the chain per element.
                    let s8_tiled = tiled_idx.is_some_and(|t| matches!(tiled[t].variant.dtype, MMADType::s8_s8_s32_s32));
                    if coned && !staged && !s8_tiled {
                        continue;
                    }
                    // A region's staged request is emitted as the direct form
                    // when either operand's view is not K-contiguous
                    // (`apply` demotes it): that is a different kernel, so the
                    // requested form must be the emitted one.
                    if let (true, Some(r)) = (staged, region)
                        && (regions[r].a_k != 1 || regions[r].b_k != 1)
                    {
                        continue;
                    }
                    if let (Some((shape, loop_len)), Some(ti)) = (&shape, tiled_idx) {
                        let t = &tiled[ti];
                        if matches!(t.variant.dtype, MMADType::s8_s8_s32_s32) {
                            // The same product as every other form: staging is
                            // an axis, and the staged layout is an axis only
                            // where there is a staged layout to choose.
                            let stages: &[(bool, &[bool])] = if staged {
                                &[(true, &[false, true])]
                            } else {
                                &[(false, &[false])]
                            };
                            for &(st, points) in stages {
                                for &swizzle in points {
                                    if s8_fit(shape, st, swizzle, layout, &info) {
                                        plans.push(Plan { region, tiled: tiled_idx, grid: None, layout, staged: st, swizzle });
                                    }
                                }
                            }
                            continue;
                        }
                        if tile_fit(shape, staged, layout, &info, |wk| self.is_sym_multiple(*loop_len, Dim::from(wk))).is_none() {
                            continue;
                        }
                    }
                    // An int8 region has no lowering of its own: the codes are
                    // read as packed words, which is the tiled cell's form, and
                    // this branch emits only the generic element-wise wmma. A
                    // region that matched an int8 form is therefore not a
                    // candidate for it.
                    if region.is_some_and(|r| matches!(regions[r].variant.dtype, MMADType::s8_s8_s32_s32)) {
                        continue;
                    }
                    // The fragment-width identity: the staged `ldmatrix` must
                    // hand the instruction exactly its operand words.
                    let variant = match (region, tiled_idx) {
                        (Some(r), _) => &regions[r].variant,
                        (None, Some(t)) => &tiled[t].variant,
                        _ => unreachable!("a plan's shape comes from a region or a tiled form"),
                    };
                    let one_step = {
                        let (fm, fn_, fk) = variant.dims.decompose_mnk();
                        TileShape {
                            mt: Dim::from(1),
                            nt: Dim::from(1),
                            kt: Dim::from(1),
                            fm: Dim::from(fm as i64),
                            fn_: Dim::from(fn_ as i64),
                            fk: Dim::from(fk as i64),
                            a_k: Dim::from(1),
                            b_k: Dim::from(1),
                            slots: Dim::from(variant.c.slots.len() as i64),
                        }
                    };
                    let widths = match &shape {
                        Some((shape, _)) => shape,
                        None => &one_step,
                    };
                    if !fragment_width_fits(variant, widths) {
                        continue;
                    }
                    // A staged layout has two points: the swizzled pack (the
                    // bank-conflict-free rows) and the plain one. An unstaged
                    // form has no staged layout to choose, so it is one point —
                    // an axis the lowering does not read is not enumerated.
                    let points: &[bool] = if staged { &[false, true] } else { &[false] };
                    for &swizzle in points {
                        plans.push(Plan { region, tiled: tiled_idx, grid: None, layout, staged, swizzle });
                    }
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

            // The operand element dtype behind a value: a plain load's storage,
            // or a fused dequant chain's value dtype.
            let elem_dtype = |k: &Kernel, o: OpId| -> Option<crate::DType> { Some(k.dtype(o)) };
            let mut ranges: Vec<(OpId, u32, OpId)> = Vec::new();
            let mut loop_id = OpId::NULL;
            let mut loop_len = OpId::NULL;
            let mut operand_dtypes: Option<(crate::DType, crate::DType)> = None;
            // The two operand *values* the contraction multiplies (a load or a
            // fused dequant chain), and the chain's int8 split when it has one.
            let mut product_values: Option<(OpId, OpId)> = None;
            // The two views the contraction multiplies: their affine maps are
            // what say which output axis each carries. A fused dequant names
            // its view through the chain's root index, exactly like a load.
            let mut product_indices: Option<(OpId, OpId)> = None;
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
                        // The product multiplies two *different* views. A
                        // dequant chain's own inner products (`code * scale`)
                        // are rooted at the very same element index, so they
                        // are never the contraction's product.
                        if let (Some((_, _, xi)), Some((_, _, yi))) = (operand_view(self, x, true), operand_view(self, y, true))
                            && xi != yi
                            && let (Some(xm), Some(ym)) = (self.get_strides(xi), self.get_strides(yi))
                            // The two views must split the output: each carries
                            // an output range the other does not. A dequant
                            // chain's own elements (a value and its group
                            // scale) name the same range on both sides.
                            && xm.keys().any(|&v| {
                                !v.is_null()
                                    && matches!(self.at(v), Op::Range { .. })
                                    && !ym.contains_key(&v)
                            })
                        {
                            operand_dtypes = elem_dtype(self, x).zip(elem_dtype(self, y));
                            product_indices = Some((xi, yi));
                            product_values = Some((x, y));
                        }
                    }
                    Op::Storage { scope: MemScope::Register, dtype, .. } => acc_dtype = Some(dtype),
                    _ => {}
                }
                op_id = self.next_op(op_id);
            }
            if std::env::var("ZYX_GD").is_ok() {
                eprintln!(
                    "GD plain={plain} ranges={} loop={} dtypes={:?} pidx={:?} acc={:?}",
                    ranges.len(),
                    loop_id.is_null(),
                    operand_dtypes,
                    product_indices,
                    acc_dtype
                );
            }
            if !ranges.is_empty()
                && !loop_id.is_null()
                && !loop_len.is_null()
                && let (Some((x_dtype, y_dtype)), Some(acc_dtype), Some((a_idx, b_idx))) =
                    (operand_dtypes, acc_dtype, product_indices)
                && let (Some(a_map), Some(b_map)) = (self.get_strides(a_idx), self.get_strides(b_idx))
                && let Some((x_val, y_val)) = product_values
            {
                // The int8 split of each operand, when it has one.
                let quants = (cone_quant(self, x_val), cone_quant(self, y_val));
                if std::env::var("ZYX_Q").is_ok() {
                    eprintln!("Q: quants={} {} acc={acc_dtype:?}", quants.0.is_some(), quants.1.is_some());
                }
                // A dequant chain is staged: reading it per fragment register
                // re-evaluates the whole chain and the device epilogue unfolds
                // each chain per fragment lane.
                let coned = operand_view(self, x_val, true).is_some_and(|(_, c, _)| c.is_some())
                    || operand_view(self, y_val, true).is_some_and(|(_, c, _)| c.is_some());
                // The two axes the instruction tiles are the ones each operand
                // carries and the other does not. Every other output axis — a
                // batch, a head — is in both views (or in neither), so it is a
                // fixed index term, not a tile axis: the block runs once per
                // instance of it. A contraction whose two operands split the
                // output axes unevenly (a direct convolution carries several)
                // has no single tile pair here and is left alone.
                let mut a_only = None;
                let mut b_only = None;
                let mut ambiguous = false;
                for &(range, _, _) in &ranges {
                    match (a_map.contains_key(&range), b_map.contains_key(&range)) {
                        (true, false) => ambiguous |= a_only.replace(range).is_some(),
                        (false, true) => ambiguous |= b_only.replace(range).is_some(),
                        _ => {}
                    }
                }
                if let (Some(m_range), Some(n_range), false) = (a_only, b_only, ambiguous) {
                    let extent = |range: OpId| ranges.iter().find(|r| r.0 == range).map(|r| r.2).unwrap_or(OpId::NULL);
                    let (m, n) = (extent(m_range), extent(n_range));
                    for variant in &info.mma_variants {
                        // The orientation is decided by the accumulator's view, not
                        // by the multiply order, so either operand may be A.
                        let s8 = matches!(variant.dtype, MMADType::s8_s8_s32_s32);
                        // A fused-dequant operand is only ever the int8 cell's:
                        // see the region gate above.
                        if coned && !s8 {
                            continue;
                        }
                        if s8 {
                            // The int8 form consumes both operands' codes and
                            // rescales their int32 sums into the region's f32
                            // accumulator: it needs both chains to expose a
                            // code and a per-group scale.
                            if quants.0.is_none() || quants.1.is_none() || acc_dtype != DType::F32 {
                                continue;
                            }
                        } else {
                            let dtypes_match = (variant.a_dtype == x_dtype && variant.b_dtype == y_dtype
                                || variant.a_dtype == y_dtype && variant.b_dtype == x_dtype)
                                && variant.c_dtype == acc_dtype;
                            if !dtypes_match {
                                continue;
                            }
                        }
                        let (fm64, fn64, fk64) = variant.dims.decompose_mnk();
                        let (fm_u, fn_u, fk_u) = (fm64 as u16, fn64 as u16, fk64 as u16);
                        let (fm, fn_, fk) = (Dim::from(fm_u), Dim::from(fn_u), Dim::from(fk_u));
                        let lane = |base: u16, mult: u16| base.saturating_mul(mult);
                        // The three splits this plan applies, filtered once per
                        // factor through split's own refusal contract: a plan whose
                        // split would be refused is not enumerated. The checks are
                        // per axis and independent of the other two, so one probe
                        // per (axis, factor) serves every combination.
                        let m_ok: Vec<bool> =
                            MULT.iter().map(|&mm| self.can_split(m_range, SplitMode::Unroll, lane(fm_u, mm))).collect();
                        let n_ok: Vec<bool> =
                            MULT.iter().map(|&nn| self.can_split(n_range, SplitMode::Unroll, lane(fn_u, nn))).collect();
                        let k_ok: Vec<bool> =
                            MULT.iter().map(|&kk| self.can_split(loop_id, SplitMode::Unroll, lane(fk_u, kk))).collect();
                        for (mi, &mm) in MULT.iter().enumerate() {
                            let m_lanes = lane(fm_u, mm);
                            if !m_ok[mi] || !self.is_sym_multiple(m, Dim::from(m_lanes)) {
                                continue;
                            }
                            for (ni, &nn) in MULT.iter().enumerate() {
                                let n_lanes = lane(fn_u, nn);
                                if !n_ok[ni] || !self.is_sym_multiple(n, Dim::from(n_lanes)) {
                                    continue;
                                }
                                for (ki, &kk) in MULT.iter().enumerate() {
                                    let k_lanes = lane(fk_u, kk);
                                    if !k_ok[ki] || !self.is_sym_multiple(loop_len, Dim::from(k_lanes)) {
                                        continue;
                                    }
                                    // A split that covers the whole K loop rewrites
                                    // the loop node into bare lanes: the reduce is
                                    // then no longer inside a loop, and the tiled
                                    // matcher emits its instructions in that loop,
                                    // so the plan has no form at all. Not a plan.
                                    if self.resolve_const(loop_len).and_then(Constant::as_dim) == Some(Dim::from(k_lanes)) {
                                        continue;
                                    }
                                    let kt = u32::from(k_lanes / fk_u);
                                    // Both operand orientations: the direct tiling
                                    // and its transpose. Each is enumerated only
                                    // when its own tile extents are whole
                                    // instructions.
                                    for transposed in [false, true] {
                                        let (mt, nt) = if transposed {
                                            if n_lanes % fm_u != 0 || m_lanes % fn_u != 0 {
                                                continue;
                                            }
                                            (u32::from(n_lanes / fm_u), u32::from(m_lanes / fn_u))
                                        } else {
                                            (u32::from(m_lanes / fm_u), u32::from(n_lanes / fn_u))
                                        };
                                        if Dim::from(i64::from(mt) * i64::from(nt) * i64::from(kt)) > MAX_TILES {
                                            continue;
                                        }
                                        // The grid as the form will see it: one
                                        // instruction tile per lane, both operands
                                        // plain K-contiguous.
                                        // The K strides the operands actually
                                        // carry: a transposed view is not
                                        // K-contiguous, and the staged form
                                        // cannot address that.
                                        let a_k = a_map.get(&loop_id).map_or(1, |&(_, s)| s);
                                        let b_k = b_map.get(&loop_id).map_or(1, |&(_, s)| s);
                                        let shape = TileShape {
                                            mt: Dim::from(mt),
                                            nt: Dim::from(nt),
                                            kt: Dim::from(kt),
                                            fm,
                                            fn_: fn_,
                                            fk,
                                            a_k,
                                            b_k,
                                            slots: Dim::from(variant.c.slots.len() as i64),
                                        };
                                        for &layout in &layouts {
                                            for staged in [false, true] {
                                                // The fragment-width identity: a staged
                                                // `ldmatrix` hands the instruction exactly
                                                // its operand words, or the device rejects
                                                // the `mma` and the slice indexes a scalar.
                                                if staged && !s8 && !fragment_width_fits(variant, &shape) {
                                                    continue;
                                                }
                                                if !staged && layout.2 > 1 {
                                                    continue;
                                                }
                                                // A dequant chain is staged: reading
                                                // it per fragment register
                                                // re-evaluates the whole chain.
                                                // The int8 cell builds each
                                                // fragment word from four code
                                                // reads, so it has an unstaged
                                                // form of its own.
                                                if coned && !staged && !s8 {
                                                    continue;
                                                }
                                                // A warp's accumulator is one register
                                                // file's worth at most: a grid whose
                                                // per-warp tile band needs more is not a
                                                // kernel the card can hold (and its
                                                // epilogue unrolls into an IR the
                                                // backend cannot render).
                                                let (wm, wn) = (Dim::from(layout.0.max(1)), Dim::from(layout.1.max(1)));
                                                if Dim::from(mt) / wm * (Dim::from(nt) / wn) * shape.slots > Dim::from(128) {
                                                    continue;
                                                }
                                                // The splits left the loop divided by
                                                // `k_lanes`: a K-split warp count
                                                // divides it when it divides the
                                                // whole loop together with them.
                                                if s8 {
                                                    let stages: &[(bool, &[bool])] = if staged {
                                                        &[(true, &[false, true])]
                                                    } else {
                                                        &[(false, &[false])]
                                                    };
                                                    for &(st, points) in stages {
                                                        for &swizzle in points {
                                                            if s8_fit(&shape, st, swizzle, layout, &info) {
                                                                plans.push(Plan {
                                                                    region: None,
                                                                    tiled: None,
                                                                    grid: Some((
                                                                        m_range, n_range, loop_id, m_lanes, n_lanes, k_lanes, fm,
                                                                        fn_, fk, transposed,
                                                                    )),
                                                                    layout,
                                                                    staged: st,
                                                                    swizzle,
                                                                });
                                                            }
                                                        }
                                                    }
                                                    continue;
                                                } else if tile_fit(&shape, staged, layout, &info, |wk| {
                                                    self.is_sym_multiple(loop_len, Dim::from(k_lanes.saturating_mul(wk)))
                                                })
                                                .is_none()
                                                {
                                                    continue;
                                                }
                                                let points: &[bool] = if staged { &[false, true] } else { &[false] };
                                                for &swizzle in points {
                                                    plans.push(Plan {
                                                        region: None,
                                                        tiled: None,
                                                        grid: Some((
                                                            m_range, n_range, loop_id, m_lanes, n_lanes, k_lanes, fm, fn_, fk,
                                                            transposed,
                                                        )),
                                                        layout,
                                                        staged,
                                                        swizzle,
                                                    });
                                                }
                                            }
                                        }
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
            name: "cuda".into(),
        }
    }

    /// The tensorization space as the enumeration built it — the same one
    /// `opt_tensorize` boxes — so a test can name a config by its plan.
    fn tensorize_space(k: &Kernel) -> Tensorize {
        k.tensorize_impl(WARP_LAYOUTS.to_vec())
    }

    /// The config of the merged tensorization space for one shape (`0` for the
    /// first matched region or pre-split tile grid) at `layout`, in the staged
    /// or the unstage form. The space offers exactly the forms each shape can
    /// be emitted in, so the config is found, never computed: a shape that
    /// cannot honour a layout has no such config.
    fn plan_config(space: &Tensorize, shape: usize, layout: (u16, u16, u16), staged: bool) -> u64 {
        let regions = space.regions.len();
        space
            .plans
            .iter()
            .position(|p| {
                let s = match (p.region, p.tiled) {
                    (Some(r), _) => r,
                    (None, Some(t)) => regions + t,
                    _ => return false,
                };
                s == shape && p.layout == layout && p.staged == staged
            })
            .unwrap_or_else(|| panic!("the space offers shape {shape} at layout {layout:?} staged={staged}")) as u64
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
        assert_eq!(
            opt.nconfigs(),
            15,
            "one region, per layout and operand form: every layout at the direct level, and the staged level only where it can be emitted — K is a single instruction step, so the only K split covers the whole loop and leaves no loop to place the instructions in, and B's view is column-major (`b_k = N`), which the staged fill cannot address, so the pass does not offer the request that `apply` would demote"
        );
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
        assert_eq!(
            opt.nconfigs(),
            51,
            "one NT region, per layout and operand form: every layout at both staging levels, and the staged level at both layouts — K is a single instruction step, so the only K split covers the whole loop and leaves no loop to place the instructions in"
        );
        opt.apply(&mut k, 0);
        k.verify();
        let wmmas = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Wmma { .. })).count();
        assert_eq!(wmmas, 1, "one wmma update");
    }

    /// A batched matmul `[b, m, k] x [b, n, k]`: the batch axis is in every
    /// view, so it is not a tile axis — the two that are (m carried by A only,
    /// n by B only) come from the views, not from the axis order, and the batch
    /// stays a fixed index term the block runs once per.
    #[test]
    fn batched_matmul_tensorizes() {
        const B: i64 = 2;
        const M: i64 = 32;
        const N: i64 = 64;
        const K: i64 = 32;

        let build = || {
            let mut k = Kernel::new(Dev::Auto);
            k.dev_info = Some(Arc::new(cuda()));
            let a = k.param(DType::F16);
            let b = k.param(DType::F16);
            let c = k.param_mut(DType::F32);
            let batch = k.group_range(0, B);
            let m = k.group_range(1, M);
            let n = k.group_range(2, N);
            let acc = k.storage(DType::F32, MemScope::Register, 1);
            let zero = k.const_val(0.0f32);
            k.store(acc, zero, 0i64);
            k.loop_over(K, |k, lv| {
                let am = k.mad(m, K, lv);
                let a_idx = k.mad(batch, M * K, am);
                let a_load = k.load(a, a_idx);
                let bn = k.mad(n, K, lv);
                let b_idx = k.mad(batch, N * K, bn);
                let b_load = k.load(b, b_idx);
                let prod = k.mul(a_load, b_load);
                let prod = k.cast(prod, DType::F32);
                let prev = k.load(acc, 0i64);
                let sum = k.add(prod, prev);
                k.store(acc, sum, 0i64);
            });
            let res = k.load(acc, 0i64);
            let mn = k.mad(m, N, n);
            let c_idx = k.mad(batch, M * N, mn);
            k.store(c, res, c_idx);
            k.linearize();
            k
        };

        let seed = build();
        let opt = seed.opt_tensorize();
        assert!(opt.nconfigs() > 0, "a batched matmul is a contraction like any other");
        // The batch axis is never a tile axis: applying a config splits m, n
        // and the contraction loop, and the batch group range stays as it is.
        // The smallest candidate is the one every other config only adds tiles
        // to, so it is the one a test can hold to a compile and a value check.
        let mut tensorized: Option<Kernel> = None;
        let mut best_n = usize::MAX;
        for cfg in 0..opt.nconfigs() {
            let mut probe = seed.clone();
            opt.apply(&mut probe, cfg);
            if probe.iter_unordered().any(|(_, op)| matches!(op, Op::Wmma { .. })) {
                let count = usize::from(probe.ops.len());
                if count < best_n {
                    best_n = count;
                    tensorized = Some(probe);
                }
            }
        }
        let Some(kernel) = tensorized else {
            panic!("no config of the batched matmul emits an instruction");
        };
        kernel.verify();

        let dev = match Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) {
            Some(d) => d,
            None => {
                eprintln!("skipping the value check: no device");
                return;
            }
        };
        let a = Tensor::rand([B, M, K], DType::F16).unwrap().to(dev).unwrap();
        let b = Tensor::rand([B, N, K], DType::F16).unwrap().to(dev).unwrap();
        let run = |k: Kernel| -> Option<Vec<f32>> {
            let compiled = k.compile().ok()?;
            let out = compiled.forward(&[&a, &b], vec![[B, M, N]]).ok()?;
            Some(out[0].clone().try_into().unwrap())
        };
        let Some(want) = run(seed.clone()) else {
            eprintln!("skipping the value check: the scalar kernel did not compile");
            return;
        };
        let Some(got) = run(kernel) else {
            panic!("the tensorized batched matmul did not compile");
        };
        let mut worst = 0.0f32;
        for i in 0..(B * M * N) as usize {
            worst = worst.max((got[i] - want[i]).abs() / (1.0 + want[i].abs()));
        }
        println!("batched matmul vs scalar: max relative |err| = {worst}");
        assert!(worst < 1e-2, "the tensorized batched matmul disagrees with the scalar kernel: {worst}");
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
        // The seed's own scalar kernel is the reference for the register
        // block below: it carries the same f16 output rounding, so a difference
        // is the tensorized form's, not the output cast's.
        let scalar = k.clone();
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

        // Every register-block tile must actually read its own lane's operand:
        // the lanes are what makes the block 16 instructions, and a lane that
        // only reaches the accumulator store leaves all 16 reading lane 0.
        let scalar = scalar.compile().expect("the scalar seed compiles");
        let want: Vec<f32> = scalar.forward(&[&a, &b], shapes()).unwrap()[1].clone().try_into().unwrap();
        let got: Vec<f32> = compiled.forward(&[&a, &b], shapes()).unwrap()[1].clone().try_into().unwrap();
        let worst = got.iter().zip(want.iter()).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        // The block sums in a different order, so the outputs differ by a few
        // f16 quanta of the result magnitude (~2^-11 relative).
        println!("split register block vs scalar seed: max |err| = {worst}");
        assert!(worst < 2.0, "the 4x4 register block disagrees with the scalar seed: {worst}");

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
        assert_eq!(
            opt.nconfigs(),
            17,
            "both operand orientations of the tiled form, at the warp layouts and staging levels that can emit them"
        );
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
        assert_eq!(opt.nconfigs(), 20, "the tiled form matches, at the warp layouts and staging levels that can emit it");
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
        let sopt = tensorize_space(&ks);
        let cfg = plan_config(&sopt, 0, WARP_LAYOUTS[0], true);
        sopt.apply(&mut ks, cfg);
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
            // The card's shared memory rather than the synthetic 8 KB: the
            // split-K form holds a warp-private tile plus the combine buffer,
            // ~9.5 KB for this shape, so an 8 KB device has no staged plan to
            // offer at any warp layout and the enumeration says so.
            let mut info = cuda();
            info.local_mem_size = 49152;
            kw.dev_info = Some(Arc::new(info));
            assert!(!kw.split(mw, split::SplitMode::Unroll, 32).is_null(), "split-K m split");
            assert!(!kw.split(nw_, split::SplitMode::Unroll, 32).is_null(), "split-K n split");
            assert!(!kw.split(klw, split::SplitMode::Unroll, 16).is_null(), "split-K k split");
            kw.verify();
            let wopt = tensorize_space(&kw);
            let cfg = plan_config(&wopt, 0, WARP_LAYOUTS[10], staged);
            wopt.apply(&mut kw, cfg);
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
        let sopt = tensorize_space(&ks);
        assert!(sopt.nconfigs() > 0, "the four-step window is a staged config");
        let cfg = plan_config(&sopt, 0, WARP_LAYOUTS[0], true);
        sopt.apply(&mut ks, cfg);
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
        let combine = plan_config(&tensorize_space(&k), 0, WARP_LAYOUTS[10], true);
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
        // One matched region, plus the tile grids the warp layouts and staging
        // levels can actually emit.
        assert_eq!(t.nconfigs(), 54, "one region and the tile grids that fit, per layout and operand form");
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
        assert_eq!(
            opt.nconfigs(),
            1623,
            "one region per layout and operand form, plus the tile grids at the warp layouts and staging levels that can emit them: this seed's B is a transposed view, so the grids are the direct form (the staged form needs both operands K-contiguous) and the direct form fits more warp layouts; the staged requests whose operand view is not K-contiguous are not offered (the emitter would demote them to the direct form)"
        );
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

    /// A fill anchored before the K loop must not clone the contraction leaf's
    /// own cone. After a split the leaf is `loop * step + lane`, so cloning its
    /// cone into the prologue (the first chunk of the loop's first iteration)
    /// puts a read of the loop ahead of the loop that defines it: a
    /// use-before-def that `compute_dtypes_and_rcs` reports as a missing dtype
    /// entry and every device refuses to render.
    #[test]
    fn cone_clone_before_the_loop_does_not_read_it() {
        const N: i64 = 8;
        const K: i64 = 32;
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cuda()));
        let w = k.param(DType::U16);
        // The prologue fill's anchor: an op that precedes the loop.
        let before = k.const_val(0i64);
        let row = k.const_val(0i64);
        let kk = k.const_val(0i64);
        let n = k.group_range(0, N);
        // The lane a split leaves in the contraction index.
        let lane = k.unroll(1, 4);
        let mut value = OpId::NULL;
        k.loop_over(K / 4, |k, lv| {
            // The split's contraction leaf: `loop * 4 + lane`, exactly the
            // shape `split` builds, so the leaf is a chain over the loop.
            let step = k.const_val(4i64);
            let stepped = k.mul(lv, step);
            let kidx = k.add(stepped, lane);
            let elem = k.mad(n, K, kidx);
            let raw = k.load(w, elem);
            value = k.cast(raw, DType::F32);
        });
        let cone = Cone::at(&k, value).expect("the dequant-style chain is one cone");
        assert_eq!(cone.mn, n, "the row leaf is the output range");
        // Rebuild the chain at an element, before the loop: what every staged
        // fill's prologue does.
        let clone = cone.clone_at_coords(&mut k, before, row, kk);
        assert!(!clone.is_null(), "the clone has a value");
        let mut loop_id = OpId::NULL;
        let mut op_id = k.head;
        while !op_id.is_null() {
            if matches!(k.at(op_id), Op::Loop { .. }) {
                loop_id = op_id;
                break;
            }
            op_id = k.next_op(op_id);
        }
        assert!(!loop_id.is_null(), "the kernel still has its K loop");
        let mut op_id = k.head;
        while op_id != loop_id {
            for p in k.at(op_id).parameters() {
                assert_ne!(p, loop_id, "an op anchored before the loop reads it: {:?}", k.at(op_id));
            }
            op_id = k.next_op(op_id);
        }
    }
}
