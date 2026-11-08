// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Local staging: deduplicate loads that threads repeat, through shared memory.
//!
//! A load whose index does not mention one of the kernel's local axes is
//! executed once per thread for the same value. Staging stores the value in
//! a `MemScope::Local` tile once, cooperatively across every local axis, so
//! the threads read it back instead of re-loading it.
//!
//! Three things make up one staged load:
//!
//! 1. **The repeat.** The load is invariant over at least one local axis;
//!    without one there is nothing to deduplicate and the load is skipped.
//! 2. **The tile.** The load's index is affine over ranges, loops and unroll
//!    lanes ([`Kernel::get_strides`]). The axes that stay inside the block —
//!    local ranges and unroll lanes — plus a strip of the innermost loop the
//!    load depends on form a dense, packed tile; the block-fixed terms
//!    (group ranges, outer loops, the constant offset) become the tile's
//!    origin. The loop is strip-mined so the body splits into a fill half
//!    and a compute half.
//! 3. **The compute.** Ops between the load and its consumers that depend
//!    only on the load and constants run in the fill, so the tile holds the
//!    value the consumers need (a dequantized element, not its packed form).
//!    The chain ends at the first op that takes another load.
//!
//! Every check runs in the make function: `apply` only executes the plan it
//! was handed, so an applied config always ends in valid IR. The pass is
//! order-agnostic — it stages whatever local axes exist and bails otherwise —
//! so a recipe (`tensorize`, `split` to lanes, then stage) is a search
//! concern, not a contract of this pass.

use super::autotune::Optimization;
use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind},
    shape::Dim,
};

/// Loop strip factors a config offers, largest first. A strip must be at
/// least as deep as the block's threads (the fill splits the tile evenly).
const STRIP_FACTORS: &[u16] = &[32, 16, 8, 4, 2];

/// Elements added to every non-innermost tile stride. One element is enough
/// to move each row off the bank it would share with the row above it: a row
/// of 32 f32 elements is exactly the 32 banks, so a strided read would hit
/// one bank 32 times.
const BANK_PAD: Dim = 1;

/// One axis of a staged tile.
#[derive(Debug, Clone)]
struct Axis {
    /// This axis is the stripped loop's own index (its tile coordinate is the
    /// inner loop variable, created in `apply`), not a variable of the index.
    loop_axis: bool,
    /// Index variable, or `OpId::NULL` for the loop axis.
    var: OpId,
    /// Number of values the tile holds along this axis.
    len: Dim,
    /// Stride of this axis in the source index.
    stride: Dim,
    /// Stride of this axis in the dense tile (packed row-major).
    compact: Dim,
}

/// A planned staging: everything `apply` needs, validated in the scan.
#[derive(Debug, Clone)]
struct StagedLoad {
    /// Buffer the load reads.
    src: OpId,
    /// The original load op, re-emitted in the fill.
    load: OpId,
    /// The value the consumers read: the last cone op, or `load` itself.
    root: OpId,
    /// Ops between `load` and `root`, in program order.
    cone: Vec<OpId>,
    /// Const-only sub-cone the cone reads, in program order: re-created in
    /// the fill, because the originals are defined in the loop body.
    consts: Vec<OpId>,
    /// The loop the tile is refilled by.
    loop_id: OpId,
    /// Strip factor: tile depth along the loop axis.
    strip: u16,
    /// Tile axes, loop axis first, then variables in op order.
    axes: Vec<Axis>,
    /// Block-fixed index terms (variable, stride), loop axis excluded.
    fixed: Vec<(OpId, Dim)>,
    /// Constant part of the source index.
    offset: Dim,
    /// Dense tile length in elements.
    tile_len: Dim,
    /// Dtype of the staged value.
    dtype: DType,
    /// Local axes (op, length) in ascending axis order: the loader's threads.
    local: Vec<(OpId, u32)>,
    /// Threads cooperating on the fill.
    threads: Dim,
}

/// Local staging optimization: one config per (repeated load, strip factor).
#[derive(Debug)]
pub struct LocalStaging {
    staged: Vec<StagedLoad>,
}

impl Optimization for LocalStaging {
    fn nconfigs(&self) -> u64 {
        self.staged.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("staging apply");
        let Some(p) = self.staged.get(config as usize) else {
            return;
        };
        let strip = Dim::from(p.strip);

        // ---- Strip-mine the loop: [Loop(len/strip), Loop(strip)], with the
        // loop variable composed as `outer * strip + inner`.
        let Op::Loop { len: len_id } = *kernel.at(p.loop_id) else {
            unreachable!("local staging: the plan's loop is no longer a loop")
        };
        let Some(len) = kernel.resolve_const(len_id).and_then(Constant::as_dim) else {
            unreachable!("local staging: the plan's loop length is no longer constant")
        };
        let outer_len = kernel.insert_const_idx_before(p.loop_id, len / strip);
        let outer = kernel.insert_before(p.loop_id, Op::Loop { len: outer_len });
        let inner_end = kernel.get_last_dim_op(p.loop_id);
        let outer_end = kernel.insert_after(inner_end, Op::EndLoop);
        let scale = kernel.insert_before(p.loop_id, Op::Const(Constant::idx(strip)));
        let scaled = kernel.insert_before(p.loop_id, Op::Binary { x: scale, y: outer, bop: BOp::Mul });
        let inner = kernel.insert_before(p.loop_id, Op::Loop { len: scale });
        kernel.ops[p.loop_id].op = Op::Binary { x: scaled, y: inner, bop: BOp::Add };

        // ---- The tile, declared before the loop that refills it.
        let smem = kernel.insert_before(outer, Op::Storage { dtype: p.dtype, scope: MemScope::Local, len: p.tile_len, id: 0 });

        // ---- Fill: cooperative copy over the product of the local axes,
        // inside the outer loop and before the inner (compute) loop.
        let iters = kernel.insert_const_idx_before(inner, p.tile_len / p.threads);
        let iter_loop = kernel.insert_before(inner, Op::Loop { len: iters });

        // Flat thread id: the hardware lane id, axis 0 fastest.
        let mut flat_lane = OpId::NULL;
        let mut stride: Dim = 1;
        for &(var, local_len) in &p.local {
            let term = if stride == 1 {
                var
            } else {
                let c = kernel.insert_before(inner, Op::Const(Constant::idx(stride)));
                kernel.insert_before(inner, Op::Binary { x: c, y: var, bop: BOp::Mul })
            };
            flat_lane = if flat_lane.is_null() {
                term
            } else {
                kernel.insert_before(inner, Op::Binary { x: flat_lane, y: term, bop: BOp::Add })
            };
            stride *= Dim::from(local_len);
        }
        let iter_stride = {
            let c = kernel.insert_before(inner, Op::Const(Constant::idx(p.threads)));
            kernel.insert_before(inner, Op::Binary { x: c, y: iter_loop, bop: BOp::Mul })
        };
        let flat = kernel.insert_before(inner, Op::Binary { x: iter_stride, y: flat_lane, bop: BOp::Add });

        // Decode the tile coordinates from the flat fill index. `flat`
        // enumerates the logical tile, so each axis divides by the *dense*
        // cumulative stride. `compact` is the padded slot stride: using it as
        // the divisor maps a flat index to a coordinate tuple whose slot is
        // not that flat index, so the fill would overwrite slots it never
        // enumerated. A single-value axis is a constant 0: leaving it as the
        // raw flat index (after the div) would build an address outside the
        // tensor.
        let mut dense: Vec<Dim> = vec![1; p.axes.len()];
        {
            let mut dense_stride: Dim = 1;
            for (i, axis) in p.axes.iter().enumerate().rev() {
                dense[i] = dense_stride;
                dense_stride *= axis.len;
            }
        }
        let mut coords: Vec<OpId> = Vec::with_capacity(p.axes.len());
        for (axis, dense) in p.axes.iter().zip(dense) {
            if axis.len == 1 {
                coords.push(kernel.insert_before(inner, Op::Const(Constant::idx(0))));
                continue;
            }
            let mut coord = flat;
            if dense != 1 {
                let c = kernel.insert_before(inner, Op::Const(Constant::idx(dense)));
                coord = kernel.insert_before(inner, Op::Binary { x: coord, y: c, bop: BOp::Div });
            }
            let c = kernel.insert_before(inner, Op::Const(Constant::idx(axis.len)));
            coord = kernel.insert_before(inner, Op::Binary { x: coord, y: c, bop: BOp::Mod });
            coords.push(coord);
        }

        // Global address: origin + every axis' contribution.
        let mut addr = kernel.insert_before(inner, Op::Const(Constant::idx(p.offset)));
        for &(var, stride) in &p.fixed {
            let s = kernel.insert_before(inner, Op::Const(Constant::idx(stride)));
            let term = kernel.insert_before(inner, Op::Binary { x: s, y: var, bop: BOp::Mul });
            addr = kernel.insert_before(inner, Op::Binary { x: addr, y: term, bop: BOp::Add });
        }
        for (axis, &coord) in p.axes.iter().zip(coords.iter()) {
            // The loop axis' coordinate is the strip-local one: the block
            // origin part is `outer * strip`.
            let coord = if axis.loop_axis {
                kernel.insert_before(inner, Op::Binary { x: scaled, y: coord, bop: BOp::Add })
            } else {
                coord
            };
            let s = kernel.insert_before(inner, Op::Const(Constant::idx(axis.stride)));
            let term = kernel.insert_before(inner, Op::Binary { x: s, y: coord, bop: BOp::Mul });
            addr = kernel.insert_before(inner, Op::Binary { x: addr, y: term, bop: BOp::Add });
        }

        // Tile slot: the axes' dense strides, with the innermost coordinate
        // rotated by its outer neighbour. The rotation is the swizzle — lanes
        // walking one row land on different banks — and the fill and the read
        // use the same formula, which keeps the mapping a bijection.
        let n_axes = p.axes.len();
        let swizzle = n_axes > 1 && p.axes[n_axes - 1].len > 1;
        let mut slot = kernel.insert_before(inner, Op::Const(Constant::idx(0)));
        for (i, axis) in p.axes.iter().enumerate() {
            let mut coord = coords[i];
            if swizzle && i + 1 == n_axes {
                let sum = kernel.insert_before(inner, Op::Binary { x: coord, y: coords[i - 1], bop: BOp::Add });
                let len = kernel.insert_before(inner, Op::Const(Constant::idx(axis.len)));
                coord = kernel.insert_before(inner, Op::Binary { x: sum, y: len, bop: BOp::Mod });
            }
            let c = kernel.insert_before(inner, Op::Const(Constant::idx(axis.compact)));
            let term = kernel.insert_before(inner, Op::Binary { x: c, y: coord, bop: BOp::Mul });
            slot = kernel.insert_before(inner, Op::Binary { x: slot, y: term, bop: BOp::Add });
        }

        // The staged value: the load plus the single-source cone above it.
        // Constants the cone reads are re-created in the fill, since their
        // originals are defined in the loop body, after the fill.
        let mut value = kernel.insert_before(inner, Op::Load { src: p.src, src_idx: addr });
        let mut clones: Map<OpId, OpId> = Map::default();
        clones.insert(p.load, value);
        for &cone_op in p.consts.iter().chain(p.cone.iter()) {
            let cloned = match *kernel.at(cone_op) {
                Op::Const(c) => kernel.insert_before(inner, Op::Const(c)),
                Op::Cast { x, dtype } => {
                    let x = *clones.get(&x).unwrap_or(&x);
                    kernel.insert_before(inner, Op::Cast { x, dtype })
                }
                Op::Bitcast { x, dtype } => {
                    let x = *clones.get(&x).unwrap_or(&x);
                    kernel.insert_before(inner, Op::Bitcast { x, dtype })
                }
                Op::Unary { x, uop } => {
                    let x = *clones.get(&x).unwrap_or(&x);
                    kernel.insert_before(inner, Op::Unary { x, uop })
                }
                Op::Binary { x, y, bop } => {
                    let x = *clones.get(&x).unwrap_or(&x);
                    let y = *clones.get(&y).unwrap_or(&y);
                    kernel.insert_before(inner, Op::Binary { x, y, bop })
                }
                _ => unreachable!("local staging: the cone is const/cast/unary/binary only"),
            };
            clones.insert(cone_op, cloned);
            value = cloned;
        }
        kernel.insert_before(inner, Op::Store { dst: smem, dst_idx: slot, src: value });
        kernel.insert_before(inner, Op::EndLoop);
        kernel.insert_before(inner, Op::Barrier);

        // ---- Compute: read the tile at this thread's coordinates, with the
        // same slot formula as the fill. Axis variables keep their own shape,
        // so a lane-unrolled axis yields a lane-vector read that `devectorize`
        // unfolds exactly like the rest of the IR.
        let mut at = inner;
        let mut idx = kernel.insert_after(at, Op::Const(Constant::idx(0)));
        at = idx;
        for (i, axis) in p.axes.iter().enumerate() {
            let coord = if axis.loop_axis { inner } else { axis.var };
            let coord = if swizzle && i + 1 == n_axes {
                let outer = &p.axes[i - 1];
                let outer = if outer.loop_axis { inner } else { outer.var };
                let sum = kernel.insert_after(at, Op::Binary { x: coord, y: outer, bop: BOp::Add });
                at = sum;
                let len = kernel.insert_after(at, Op::Const(Constant::idx(axis.len)));
                at = len;
                let rotated = kernel.insert_after(at, Op::Binary { x: sum, y: len, bop: BOp::Mod });
                at = rotated;
                rotated
            } else {
                coord
            };
            let c = kernel.insert_after(at, Op::Const(Constant::idx(axis.compact)));
            at = c;
            let term = kernel.insert_after(at, Op::Binary { x: c, y: coord, bop: BOp::Mul });
            at = term;
            let sum = kernel.insert_after(at, Op::Binary { x: term, y: idx, bop: BOp::Add });
            at = sum;
            idx = sum;
        }
        let staged_load = kernel.insert_after(at, Op::Load { src: smem, src_idx: idx });

        // Every consumer of the root now reads the tile.
        kernel.remap(p.root, staged_load);

        // The next fill must not overwrite the tile while it is still read.
        kernel.insert_before(outer_end, Op::Barrier);

        kernel.verify();
    }
}

impl Kernel {
    /// Make the local staging optimization: scan for loads that threads
    /// repeat and plan a shared-memory tile for each. Never mutates.
    ///
    /// One config per (repeated load, strip factor). A config bails when the
    /// device has no shared memory, the kernel has no local axis, the load's
    /// index is not affine, no counted loop carries the load, the tile does
    /// not fit in the remaining local memory, or the tile cannot be split
    /// evenly over the block's threads.
    pub fn opt_local_staging(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_local_staging");
        let mut staged: Vec<StagedLoad> = Vec::new();

        let info = self.device_info();
        if info.local_mem_size <= 0 {
            return Box::new(LocalStaging { staged });
        }

        // Local axes, one entry per axis, ascending: the hardware threads.
        let mut local: Vec<(u32, OpId, u32)> = Vec::new();
        for (op_id, op) in self.iter_unordered() {
            if let Op::Range { axis, kind: RangeKind::Local(len) } = *op
                && !local.iter().any(|&(a, _, _)| a == axis)
            {
                local.push((axis, op_id, len));
            }
        }
        local.sort_by_key(|&(axis, _, _)| axis);
        if local.is_empty() {
            return Box::new(LocalStaging { staged });
        }
        let threads: Dim = local.iter().map(|&(_, _, len)| Dim::from(len)).product();
        if threads <= 0 {
            return Box::new(LocalStaging { staged });
        }

        // Shared memory already spoken for.
        let mut used: Dim = 0;
        for (_, op) in self.iter_unordered() {
            if let Op::Storage { dtype, scope: MemScope::Local, len, .. } = *op {
                used += len * Dim::from(dtype.bit_size() / 8);
            }
        }
        let budget = info.local_mem_size - used;
        if budget <= 0 {
            return Box::new(LocalStaging { staged });
        }

        // Program order positions and a consumer map, once for the scan.
        let mut pos: Map<OpId, usize> = Map::default();
        let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
        let mut walk_id = self.head;
        let mut n = 0usize;
        while !walk_id.is_null() {
            pos.insert(walk_id, n);
            for p in self.at(walk_id).parameters() {
                if !p.is_null() {
                    consumers.entry(p).or_default().push(walk_id);
                }
            }
            n += 1;
            walk_id = self.next_op(walk_id);
        }
        let inside = |op_id: OpId, loop_id: OpId, end: OpId| -> bool {
            match (pos.get(&op_id), pos.get(&loop_id), pos.get(&end)) {
                (Some(&o), Some(&l), Some(&e)) => l < o && o < e,
                _ => false,
            }
        };
        // Const-only expression: holds no load and no variable.
        let is_const_expr = |op_id: OpId| -> bool {
            let mut stack = vec![op_id];
            for _ in 0..64 {
                let Some(x) = stack.pop() else { return true };
                match self.at(x) {
                    Op::Const(_) => {}
                    Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Unary { x, .. } => stack.push(*x),
                    Op::Binary { x, y, .. } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    _ => return false,
                }
            }
            false
        };

        // Program-order scan: the open loop nest is the context of a load.
        let mut open: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Loop { .. } => open.push(op_id),
                Op::EndLoop => {
                    open.pop();
                }
                Op::Load { .. } => {
                    let Op::Load { src, src_idx } = *self.at(op_id) else {
                        unreachable!()
                    };
                    if !matches!(self.at(src), Op::Param { kind: ParamKind::Global, .. }) {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    let Some(map) = self.get_strides(src_idx) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };

                    // The repeat: at least one local axis the load does not
                    // depend on. Without one there is nothing to deduplicate.
                    if local.iter().all(|&(_, var, _)| map.contains_key(&var)) {
                        op_id = self.next_op(op_id);
                        continue;
                    }

                    // The innermost open loop the index depends on, counted.
                    let Some(&loop_id) = open.iter().rev().find(|&&l| map.contains_key(&l)) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    let Op::Loop { len: loop_len } = *self.at(loop_id) else {
                        unreachable!()
                    };
                    let Some(len) = self.resolve_const(loop_len).and_then(Constant::as_dim) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    if len < 2 {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    let loop_end = self.get_last_dim_op(loop_id);

                    // A barrier in the body would pin the fill/compute split.
                    {
                        let mut scan = self.next_op(loop_id);
                        let mut barred = false;
                        while scan != loop_end {
                            if matches!(self.at(scan), Op::Barrier) {
                                barred = true;
                                break;
                            }
                            scan = self.next_op(scan);
                        }
                        if barred {
                            op_id = self.next_op(op_id);
                            continue;
                        }
                    }

                    // Tile axes and the block-fixed origin terms.
                    let mut tile: Vec<(OpId, Dim, Dim)> = Vec::new();
                    let mut fixed: Vec<(OpId, Dim)> = Vec::new();
                    let mut offset: Dim = 0;
                    let mut skip = false;
                    for (&var, &(_var_len, stride)) in map.iter() {
                        if var.is_null() {
                            offset = stride;
                            continue;
                        }
                        if var == loop_id {
                            continue;
                        }
                        match *self.at(var) {
                            Op::Range { kind: RangeKind::Local(len), .. } => {
                                tile.push((var, Dim::from(len), stride));
                            }
                            Op::Unroll { len, .. } => tile.push((var, Dim::from(len), stride)),
                            Op::Range { .. } | Op::Loop { .. } => fixed.push((var, stride)),
                            _ => skip = true,
                        }
                    }
                    if skip {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    tile.sort_by_key(|&(var, _, _)| var);
                    let loop_stride = map[&loop_id].1;

                    // The single-source cone: cast/unary/binary above the
                    // load, ending at the first op that takes another load.
                    let mut cone: Vec<OpId> = Vec::new();
                    let mut root = op_id;
                    let mut consumable = true;
                    loop {
                        let Some(next) = consumers.get(&root) else {
                            consumable = false;
                            break;
                        };
                        if next.len() != 1 {
                            break;
                        }
                        let c = next[0];
                        if !inside(c, loop_id, loop_end) {
                            consumable = false;
                            break;
                        }
                        let advanced = match *self.at(c) {
                            Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Unary { x, .. } if x == root => {
                                cone.push(c);
                                root = c;
                                true
                            }
                            Op::Binary { x, y, .. } if (x == root) != (y == root) => {
                                let other = if x == root { y } else { x };
                                if is_const_expr(other) {
                                    cone.push(c);
                                    root = c;
                                    true
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        };
                        if !advanced {
                            break;
                        }
                    }
                    if !consumable {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    // Every consumer of the staged value must be inside the
                    // refilled loop body: outside it the tile is stale.
                    let Some(root_consumers) = consumers.get(&root) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    if !root_consumers.iter().all(|&c| inside(c, loop_id, loop_end)) {
                        op_id = self.next_op(op_id);
                        continue;
                    }

                    let dtype = self.dtype(root);

                    // Const-only sub-cone of the cone's operands, in program
                    // order: the fill re-creates it next to the load.
                    let mut const_set: crate::Set<OpId> = crate::Set::default();
                    let mut stack: Vec<OpId> = Vec::new();
                    for &c in &cone {
                        for p in self.at(c).parameters() {
                            if p.is_null() || p == op_id || cone.contains(&p) {
                                continue;
                            }
                            stack.push(p);
                        }
                    }
                    while let Some(x) = stack.pop() {
                        if !const_set.insert(x) {
                            continue;
                        }
                        for p in self.at(x).parameters() {
                            if !p.is_null() && p != op_id && !cone.contains(&p) {
                                stack.push(p);
                            }
                        }
                    }
                    let mut consts: Vec<OpId> = const_set.into_iter().collect();
                    consts.sort_by_key(|&id| pos[&id]);

                    for &factor in STRIP_FACTORS {
                        let strip = Dim::from(factor);
                        if len % strip != 0 {
                            continue;
                        }

                        // Loop axis first, then the index variables in op
                        // order: one dense layout for fill and read. Every
                        // non-innermost stride carries `BANK_PAD` extra
                        // elements so consecutive rows do not land on the
                        // same bank, and the tile is rounded up to the
                        // block's thread count so the fill stays even.
                        let mut axes =
                            vec![Axis { loop_axis: true, var: OpId::NULL, len: strip, stride: loop_stride, compact: 1 }];
                        for &(var, axis_len, stride) in &tile {
                            axes.push(Axis { loop_axis: false, var, len: axis_len, stride, compact: 1 });
                        }
                        let pad = if axes.len() > 1 { BANK_PAD } else { 0 };
                        let mut stride: Dim = 1;
                        for axis in axes.iter_mut().rev() {
                            axis.compact = stride;
                            stride = axis.len * stride + pad;
                        }
                        let span = stride - pad;
                        let tile_len = (span + threads - 1) / threads * threads;
                        if tile_len * Dim::from(dtype.bit_size() / 8) > budget {
                            continue;
                        }

                        staged.push(StagedLoad {
                            src,
                            load: op_id,
                            root,
                            cone: cone.clone(),
                            consts: consts.clone(),
                            loop_id,
                            strip: factor,
                            axes,
                            fixed: fixed.clone(),
                            offset,
                            tile_len,
                            dtype,
                            local: local.iter().map(|&(_, var, len)| (var, len)).collect(),
                            threads,
                        });
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        Box::new(LocalStaging { staged })
    }
}

/// One MMA fragment staged through shared memory.
#[derive(Debug, Clone)]
struct MmaStage {
    /// The `Op::Wmma` whose operand is staged.
    wmma: OpId,
    /// `true`: the A operand; `false`: B.
    a_side: bool,
    /// The kernel's local (lane) range.
    lane: OpId,
    /// The operand's fragment map, cloned from the device variant.
    fields: Vec<crate::backend::FragField>,
    /// `(row offset, col offset, slot load)` per fragment slot.
    slots: Vec<(i16, i16, OpId)>,
    /// Fragment rows and contraction width.
    rows: Dim,
    k: Dim,
}

/// Staging for tensor-core fragments: copy the A/B tile each warp needs into
/// shared memory once, cooperatively, so the fragment slots read it back
/// instead of issuing lane-scattered global loads. A following peephole can
/// then fold those reads into `ldmatrix`.
#[derive(Debug, Clone)]
pub struct MmaStaging {
    stages: Vec<MmaStage>,
}

/// Build one fragment axis from the lane bit-fields: sum of
/// `((lane >> shift) & mask) * stride` over the fields feeding `axis`.
fn lane_axis(kernel: &mut Kernel, before: OpId, lane: OpId, fields: &[crate::backend::FragField], axis: crate::backend::FragAxis) -> OpId {
    let mut sum = kernel.insert_before(before, Op::Const(Constant::idx(0u32)));
    for f in fields.iter().filter(|f| f.axis == axis) {
        let mut term = lane;
        if f.shift > 0 {
            let d = kernel.insert_before(before, Op::Const(Constant::idx(1u32 << f.shift)));
            term = kernel.insert_before(before, Op::Binary { x: term, y: d, bop: BOp::Div });
        }
        let mask = (1u32 << f.width) - 1;
        let m = kernel.insert_before(before, Op::Const(Constant::idx(mask + 1)));
        term = kernel.insert_before(before, Op::Binary { x: term, y: m, bop: BOp::Mod });
        if f.stride != 1 {
            let s = kernel.insert_before(before, Op::Const(Constant::idx(f.stride as u32)));
            term = kernel.insert_before(before, Op::Binary { x: term, y: s, bop: BOp::Mul });
        }
        sum = kernel.insert_before(before, Op::Binary { x: sum, y: term, bop: BOp::Add });
    }
    sum
}

impl Optimization for MmaStaging {
    fn nconfigs(&self) -> u64 {
        self.stages.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        let Some(st) = self.stages.get(config as usize) else {
            return;
        };
        let wmma = st.wmma;
        let head = kernel.head;

        // One shared tile per fragment, row-major with `k` elements per row.
        // Inserted after the params: global params must precede local
        // storages in the kernel.
        let mut at = head;
        while matches!(kernel.at(at), Op::Param { .. }) {
            at = kernel.next_op(at);
        }
        let buf = kernel.insert_before(
            at,
            Op::Storage { dtype: DType::F16, scope: MemScope::Local, len: st.rows * st.k, id: 0 },
        );

        let row = lane_axis(kernel, wmma, st.lane, &st.fields, crate::backend::FragAxis::Row);
        let col = lane_axis(kernel, wmma, st.lane, &st.fields, crate::backend::FragAxis::Col);
        let k_const = kernel.insert_before(wmma, Op::Const(Constant::idx(st.k as u32)));
        let base = kernel.insert_before(wmma, Op::Binary { x: row, y: k_const, bop: BOp::Mul });
        let base = kernel.insert_before(wmma, Op::Binary { x: base, y: col, bop: BOp::Add });

        // Fill: every lane stores its own fragment elements at their tile
        // positions, then a barrier makes the tile whole for the reads.
        let mut positions = Vec::with_capacity(st.slots.len());
        for &(roff, coff, src) in &st.slots {
            // One combined offset keeps the index tree to `base [+ const]`.
            let off = Dim::from(roff) * st.k + Dim::from(coff);
            let mut pos = base;
            if off != 0 {
                let c = kernel.insert_before(wmma, Op::Const(Constant::idx(off as u32)));
                pos = kernel.insert_before(wmma, Op::Binary { x: pos, y: c, bop: BOp::Add });
            }
            kernel.insert_before(wmma, Op::Store { dst: buf, dst_idx: pos, src });
            positions.push(pos);
        }
        kernel.insert_before(wmma, Op::Barrier);

        // Compute: the fragment reads the staged tile. The old stack and its
        // global loads stay in place; the operand repoint makes them dead and
        // DCE removes what is unused.
        let mut staged = Vec::with_capacity(positions.len());
        for &pos in &positions {
            staged.push(kernel.insert_before(wmma, Op::Load { src: buf, src_idx: pos }));
        }
        let stack = kernel.insert_before(wmma, Op::Stack { ops: staged.into() });
        match &mut kernel.ops[wmma].op {
            Op::Wmma { a, b, .. } => {
                if st.a_side {
                    *a = stack;
                } else {
                    *b = stack;
                }
            }
            _ => unreachable!("MmaStaging: config is not a wmma"),
        }
    }
}

impl Kernel {
    /// Stage MMA fragments in shared memory, one config per (wmma, operand).
    ///
    /// The matcher is structural: it binds the operand stack, checks that
    /// every slot is a load of the same global buffer, and derives the slot
    /// positions from the device variant's [`crate::backend::FragMap`] rather
    /// than from the (epilogue-simplified) index arithmetic.
    pub fn opt_mma_staging(&self) -> Box<dyn Optimization> {
        let info = self.device_info();
        let mut stages = Vec::new();
        if info.mma_variants.is_empty() {
            return Box::new(MmaStaging { stages });
        }
        let lane = self.iter_unordered().find_map(|(id, op)| {
            matches!(op, Op::Range { kind: RangeKind::Local(_), .. }).then_some(id)
        });
        let Some(lane) = lane else {
            return Box::new(MmaStaging { stages });
        };
        for (wmma, op) in self.iter_unordered() {
            let Op::Wmma { dims, layout, dtype, a, b, .. } = *op else {
                continue;
            };
            let Some(v) = info
                .mma_variants
                .iter()
                .find(|v| v.dims == dims && v.layout == layout && v.dtype == dtype)
            else {
                continue;
            };
            let (fm, fn_, fk) = dims.decompose_mnk();
            for (a_side, operand, map, rows) in [(true, a, &v.a, fm), (false, b, &v.b, fn_)] {
                let Op::Stack { ref ops } = *self.at(operand) else {
                    continue;
                };
                if ops.len() != map.slots.len() {
                    continue;
                }
                let mut buf = OpId::NULL;
                let mut slots = Vec::with_capacity(ops.len());
                let mut ok = true;
                for (i, &slot_op) in ops.iter().enumerate() {
                    let Op::Load { src, .. } = *self.at(slot_op) else {
                        ok = false;
                        break;
                    };
                    if !matches!(self.at(src), Op::Param { kind: ParamKind::Global, .. }) {
                        ok = false;
                        break;
                    }
                    if buf.is_null() {
                        buf = src;
                    } else if buf != src {
                        ok = false;
                        break;
                    }
                    slots.push((map.slots[i].0, map.slots[i].1, slot_op));
                }
                if !ok || buf.is_null() {
                    continue;
                }
                stages.push(MmaStage {
                    wmma,
                    a_side,
                    lane,
                    fields: map.fields.clone(),
                    slots,
                    rows: Dim::from(rows as Dim),
                    k: Dim::from(fk as Dim),
                });
            }
        }
        Box::new(MmaStaging { stages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{DTypeCapability, DeviceInfo},
        kernel::{Dev, split},
    };
    use std::sync::Arc;

    fn cpu() -> DeviceInfo {
        DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            mma_variants: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }
    }

    /// A 16-trip reduction whose input load is invariant over the 4-thread
    /// lane: every thread loads the same element, so the tile holds the
    /// strip's elements once and the consumers read them back.
    #[test]
    fn stages_a_repeated_load() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let _lane = k.local_range(0, 4);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(16i64);
        k.loop_over(16i64, |k, kv| {
            let idx = k.mad(g, row, kv);
            let x = k.load(a, idx);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        let opt = k.opt_local_staging();
        // Every strip factor of the 16-trip loop; the tile is rounded up to
        // the block's thread count, so a 2-element strip still fills evenly.
        assert_eq!(opt.nconfigs(), 4, "one config per strip factor");
        opt.apply(&mut k, 0);
        k.verify();

        let locals: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Storage { scope: MemScope::Local, len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        assert_eq!(locals, vec![16], "one tile: the strip, f32");
        assert!(k.iter_unordered().any(|(_, op)| matches!(op, Op::Barrier)), "fill and read are separated");
        let smem = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Storage { scope: MemScope::Local, .. }).then_some(id))
            .expect("the staged tile");
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Load { src, .. } if *src == smem)),
            "the consumer reads the tile"
        );
    }

    /// Without a local axis there are no threads to deduplicate across.
    #[test]
    fn bails_without_local_axis() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(16i64);
        k.loop_over(16i64, |k, kv| {
            let idx = k.mad(g, row, kv);
            let x = k.load(a, idx);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        assert_eq!(k.opt_local_staging().nconfigs(), 0, "no lane, no sharing");
    }

    /// The staged value's cone ends at the first op that takes another load:
    /// the multiply of the two operands stays outside the tile.
    #[test]
    fn cone_ends_at_a_second_load() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let lane = k.local_range(0, 4);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(16i64);
        k.loop_over(16i64, |k, kv| {
            let a_idx = k.mad(g, row, kv);
            let b_row = k.mul(lane, row);
            let b_idx = k.add(b_row, kv);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        let opt = k.opt_local_staging();
        assert!(opt.nconfigs() > 0, "the lane-invariant load is stageable");
        opt.apply(&mut k, 0);
        k.verify();

        let stored = k
            .iter_unordered()
            .find_map(|(_, op)| match op {
                Op::Store { dst, src, .. } if matches!(k.at(*dst), Op::Storage { scope: MemScope::Local, .. }) => Some(*src),
                _ => None,
            })
            .expect("the fill stores one value");
        assert!(matches!(k.at(stored), Op::Load { .. }), "the second load ends the cone, so the tile holds the raw element");
    }

    /// A symbolic loop bound leaves no constant tile extent.
    #[test]
    fn bails_when_an_extent_is_symbolic() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let _lane = k.local_range(0, 4);
        let n = k.variable(DType::I64);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(16i64);
        k.loop_over(n, |k, kv| {
            let idx = k.mad(g, row, kv);
            let x = k.load(a, idx);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        assert_eq!(k.opt_local_staging().nconfigs(), 0, "no constant extent, no tile");
    }

    /// Compute that depends only on the load and constants runs in the fill:
    /// the tile holds the converted value, not the raw element.
    #[test]
    fn stages_the_post_load_compute() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F16);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let _lane = k.local_range(0, 4);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(16i64);
        k.loop_over(16i64, |k, kv| {
            let idx = k.mad(g, row, kv);
            let x = k.load(a, idx);
            let y = k.cast(x, DType::F32);
            let scale = k.const_val(2.0f32);
            let z = k.mul(y, scale);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, z);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        let opt = k.opt_local_staging();
        assert!(opt.nconfigs() > 0);
        opt.apply(&mut k, 0);
        k.verify();

        let (stored, dtype) = k
            .iter_unordered()
            .find_map(|(_, op)| match op {
                Op::Store { dst, src, .. } => match k.at(*dst) {
                    Op::Storage { dtype, scope: MemScope::Local, .. } => Some((*src, *dtype)),
                    _ => None,
                },
                _ => None,
            })
            .expect("the fill stores one value");
        assert_eq!(dtype, DType::F32, "the tile holds the converted value");
        assert!(matches!(k.at(stored), Op::Binary { bop: BOp::Mul, .. }), "the scale is applied once per element, in the fill");
    }

    /// The loop is strip-mined: the tile is one strip deep and the body is
    /// split into a fill half and a compute half.
    #[test]
    fn strip_splits_the_loop() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let _lane = k.local_range(0, 8);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(64i64);
        k.loop_over(64i64, |k, kv| {
            let idx = k.mad(g, row, kv);
            let x = k.load(a, idx);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        k.store(out, r, g);

        let opt = k.opt_local_staging();
        assert!(opt.nconfigs() > 0);
        opt.apply(&mut k, 0);
        k.verify();

        let loops: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Loop { len } => k.resolve_const(*len).and_then(Constant::as_dim),
                _ => None,
            })
            .collect();
        assert!(loops.contains(&2), "64 trips / 32 = the outer loop");
        assert!(loops.contains(&32), "the strip is a loop of its own");
        let locals: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Storage { scope: MemScope::Local, len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        assert_eq!(locals, vec![32], "one tile per strip, one element per trip");
    }

    /// The recipe's middle step: lanes first, then staging. The M operand is
    /// invariant over the N lane, so it is shared; the N operand varies with
    /// it and is left alone.
    #[test]
    fn stages_a_split_matmul() {
        const M: i64 = 32;
        const N: i64 = 32;
        const K: Dim = 32;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, kv| {
            let a_idx = k.mad(m, K, kv);
            let b_idx = k.mad(kv, N, n);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, r, c_idx);

        let n_range = k
            .iter_unordered()
            .find_map(|(id, op)| match *op {
                Op::Range { axis: 1, kind: RangeKind::Group(_) } => Some(id),
                _ => None,
            })
            .expect("the N group range");
        let split_opt = split::OptSplit { configs: vec![(n_range, split::SplitMode::Local, 4)] };
        Optimization::apply(&split_opt, &mut k, 0);
        k.verify();

        let opt = k.opt_local_staging();
        assert!(opt.nconfigs() > 0, "the M operand is lane-invariant");
        opt.apply(&mut k, 0);
        k.verify();

        let locals: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Storage { scope: MemScope::Local, len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        assert_eq!(locals, vec![32], "the K loop's strip, 32 elements");
    }

    /// A tile with an inner axis: the row stride carries one element of bank
    /// padding and the innermost coordinate is rotated by its outer
    /// neighbour, so lanes walking a row spread over the banks.
    #[test]
    fn pads_and_swizzles_the_tile() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 8);
        let _lane0 = k.local_range(0, 2);
        let lane1 = k.local_range(1, 4);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        let row = k.const_idx(64i64);
        let tile_row = k.const_idx(8i64);
        k.loop_over(8i64, |k, kv| {
            // Depends on lane1 and the loop, invariant over lane0.
            let inner = k.mad(lane1, tile_row, kv);
            let idx = k.mad(g, row, inner);
            let x = k.load(a, idx);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prev, x);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(g, 8i64, lane1);
        k.store(out, r, c_idx);

        let opt = k.opt_local_staging();
        assert!(opt.nconfigs() > 0, "lane0 does not index the load");
        // Config 0 is the 8-trip strip.
        opt.apply(&mut k, 0);
        k.verify();

        let locals: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Storage { scope: MemScope::Local, len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        // Packed would be 8 * 4 = 32; one element of padding per row of the
        // 4-wide inner axis makes the span 40, rounded up to 8 threads.
        assert_eq!(locals, vec![40], "a padded 8 x 4 tile");
        let smem = k
            .iter_unordered()
            .find_map(|(id, op)| matches!(op, Op::Storage { scope: MemScope::Local, .. }).then_some(id))
            .expect("the staged tile");
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Load { src, .. } if *src == smem)),
            "the consumer reads the tile"
        );
    }
}
