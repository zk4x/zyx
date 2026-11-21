// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tiled parallel reduction for large single-dimension reduction kernels.
//!
//! This optimization targets kernels that contain a **single large loop reducing
//! into one register accumulator** — e.g., `for i in 0..32000 { sum += data[i] }`.
//! It parallelizes the reduction across threads by splitting the loop iterations,
//! storing partial sums in shared memory, and doing a tree-reduce to produce the
//! final scalar result.
//!
//! **This is NOT a matmul optimization.** Applying it to a matmul kernel
//! would make performance worse than a naive matmul. It is designed for
//! standalone reduction ops (e.g. `Tensor::sum` over a large axis).

use super::autotune::Optimization;
use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind},
    shape::Dim,
};

/// Tiled reduction parallelization.
#[derive(Debug)]
pub struct TiledReduce {
    /// Tuples of (operation_id, local_factor, global_factor) for each tiled reduce.
    pub factors: Vec<(OpId, u64, u64)>,
    /// The int8 vec-dot form: one entry per matched group-reduction loop, the
    /// thread splits it is offered with (`32`, `64`, `128`).
    pub vec_dots: Vec<(OpId, u32)>,
}

impl Optimization for TiledReduce {
    fn nconfigs(&self) -> u64 {
        (self.factors.len() + self.vec_dots.len()) as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        // See `PadIndex::apply`: an id that is not in this instance's list
        // means the config is unavailable on the state being replayed.
        let config = config as usize;
        if let Some(&(op_id, factor, tree_branch)) = self.factors.get(config) {
            kernel.local_reduce(op_id, factor as u32, tree_branch as u32);
            return;
        }
        if let Some(&(op_id, threads)) = self.vec_dots.get(config - self.factors.len()) {
            kernel.vec_dot_split(op_id, threads);
        }
    }
}

impl Kernel {
    /// Make the `TiledReduce` optimization: scan the kernel for reduction
    /// loops that can be parallelized across threads.
    pub fn opt_local_reduce(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_tiled_reduce");

        let dev_info = self.dev_info();

        // The int8 vec-dot form is matched first: it does not need the
        // single-loop skeleton the general split requires, and it replaces
        // the body it finds. Both rewrites allocate local axes and a barrier,
        // so only a kernel with neither is offered a config.
        let clean = !self.ops.values().any(|node| matches!(node.op, Op::Barrier | Op::Range { kind: RangeKind::Local(_), .. }));
        let mut vec_dots = Vec::new();
        if clean {
            for (id, node) in self.ops.iter() {
                if matches!(node.op, Op::Loop { .. }) && self.vec_dot_shape(id).is_some() {
                    for split in [128u32, 64, 32] {
                        if split <= dev_info.max_local_threads {
                            vec_dots.push((id, split));
                        }
                    }
                }
            }
        }

        // Let's not tile reduce kernel with barriers for now
        // Don't apply tiled reduce if there's already a barrier or local index
        if !clean {
            return Box::new(TiledReduce { factors: Vec::new(), vec_dots });
        }
        // Only apply tiled reduce if there's exactly one loop in the kernel
        let n_loops = self.ops.values().filter(|node| matches!(node.op, Op::Loop { .. })).count();
        if n_loops != 1 {
            return Box::new(TiledReduce { factors: Vec::new(), vec_dots });
        }

        let mut local_axis_sizes: Map<u32, u32> = crate::Map::default();
        for op in self.ops.values() {
            if let Op::Range { axis, kind: RangeKind::Local(len) } = op.op {
                if let Some(&existing) = local_axis_sizes.get(&axis) {
                    debug_assert_eq!(existing, len);
                } else {
                    local_axis_sizes.insert(axis, len);
                }
            }
        }
        let used_threads: u32 = local_axis_sizes.values().product();
        let remaining_threads = if local_axis_sizes.is_empty() {
            dev_info.max_local_threads
        } else {
            dev_info.max_local_threads / used_threads
        };

        let candidates = vec![1024, 512, 256, 128, 64, 32, 16, 8];
        let tree_branch_candidates = vec![2, 4, 8, 16];
        let mut factors = Vec::new();
        // A tiled reduce splits the loop into `factor` lanes and stages the
        // tree over them: the emitted IR grows with the factor, so refuse a
        // config that would take the kernel past what the search can afford
        // (the same bound `opt_split` and `opt_register_tiling` use).
        const SEARCH_OPS: u64 = 10_000;
        let op_len: u64 = self.ops.len().0 as u64;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Loop { len: len_id } = self.ops[op_id].op {
                let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) else {
                    op_id = next;
                    continue;
                };
                if len >= 16 {
                    for &factor in &candidates {
                        if len % factor as Dim == 0
                            && len / factor as Dim >= 4
                            && remaining_threads as Dim >= factor as Dim
                            && op_len.saturating_mul(u64::from(factor)) <= SEARCH_OPS
                        {
                            for &tree_branch in &tree_branch_candidates {
                                factors.push((op_id, factor, tree_branch));
                            }
                        }
                    }
                }
            }
            op_id = next;
        }
        Box::new(TiledReduce { factors, vec_dots })
    }

    /// Apply tiled reduction parallelization.
    ///
    /// This method parallelizes a large single-dimension reduction loop
    /// across threads by splitting the loop iterations and storing
    /// partial sums in shared memory, then performing a tree reduction.
    ///
    /// # Arguments
    ///
    /// * `loop_start` - The loop operation to parallelize
    /// * `factor` - The factor for splitting the loop
    /// * `tree_branch` - The tree reduction branching factor
    pub(crate) fn local_reduce(&mut self, loop_start: OpId, factor: u32, tree_branch: u32) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("tiled_reduce");
        let loop_len_id = if let Op::Loop { len } = self.at(loop_start) {
            *len
        } else {
            return;
        };
        let Some(loop_len) = self.resolve_const(loop_len_id).and_then(crate::dtype::Constant::as_dim) else {
            return;
        };

        // Get new free axis for the local dimension
        let laxis = self
            .ops
            .values()
            .filter_map(|node| {
                if let Op::Range { axis, kind: RangeKind::Local(_), .. } = node.op {
                    Some(axis + 1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        if laxis > 2 {
            return;
        }

        // Find the acc definition
        let mut op_id = loop_start;
        let reg_acc;
        let acc_dtype;
        loop {
            if let Op::Storage { dtype, scope, len, .. } = self.ops[op_id].op {
                if scope != MemScope::Register || len != 1 {
                    return;
                }
                reg_acc = op_id;
                acc_dtype = dtype;
                break;
            }
            op_id = self.prev_op(op_id);
            if op_id == OpId::NULL {
                // Accumulator was no found
                return;
            }
        }
        debug_assert!(!reg_acc.is_null());
        // The tree stages `factor` accumulator elements in thread-shared
        // local memory; refuse when they do not fit. This also excludes
        // devices with no thread-shared local memory at all.
        let required = factor as Dim * Dim::from(acc_dtype.bit_size() / 8);
        if required > self.dev_info().local_mem_size {
            return;
        }

        // Find the reduce loop bop and the op that used to load from the register accumulator
        let mut reduce_bop_id = OpId::NULL;
        let acc_load_id;
        let mut op_id = self.next_op(loop_start);
        let mut depth = 1;
        loop {
            match self.ops[op_id].op {
                // Update store to use the lidx for indexing
                Op::Store { dst, src: x, .. } => {
                    if dst == reg_acc {
                        reduce_bop_id = x;
                    }
                }
                Op::Copy { src, dst, .. } => {
                    if src == reg_acc || dst == reg_acc {
                        return;
                    }
                }
                Op::Load { src, .. } if depth == 0 => {
                    if src == reg_acc {
                        acc_load_id = op_id;
                        break;
                    }
                }
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => depth -= 1,
                _ => {}
            }
            op_id = self.next_op(op_id);
            if op_id.is_null() {
                return;
            }
        }
        debug_assert!(!reduce_bop_id.is_null());
        let Op::Binary { bop, .. } = self.ops[reduce_bop_id].op else {
            return;
        };

        // ***** IMPLEMENTATION ***** //

        // Find the last global param to insert local memory after it
        let mut last_global = None;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Param { .. }) {
                last_global = Some(op_id);
            }
            op_id = self.next_op(op_id);
        }

        // Insert local memory storages right after the last global param
        let insert_at = match last_global {
            Some(g) => {
                let n = self.next_op(g);
                if n.is_null() { self.tail } else { n }
            }
            None => self.head,
        };
        let loc_acc =
            self.insert_before(insert_at, Op::Storage { dtype: acc_dtype, scope: MemScope::Local, len: factor as Dim, id: 0 });
        let lidx = self.insert_before(insert_at, Op::Range { axis: laxis, kind: RangeKind::Local(factor) });

        // Divide reduce loop by factor
        let factor_const = self.insert_before(loop_start, Op::Const(Constant::idx(factor)));
        let new_len = self.insert_const_idx_before(loop_start, loop_len / factor as Dim);
        let ridx = self.insert_before(loop_start, Op::Loop { len: new_len });
        let xy = self.insert_before(loop_start, Op::Binary { x: ridx, y: factor_const, bop: BOp::Mul });
        self.ops[loop_start].op = Op::Binary { x: xy, y: lidx, bop: BOp::Add };

        // Store to local accumulator
        let const_zero = self.insert_before(acc_load_id, Op::Const(Constant::idx(0)));
        let x = self.insert_before(acc_load_id, Op::Load { src: reg_acc, src_idx: const_zero });
        self.insert_before(acc_load_id, Op::Store { dst: loc_acc, dst_idx: lidx, src: x });

        // Sync memory
        self.insert_before(acc_load_id, Op::Barrier);

        // Tree reduce: each step threads with lidx < active_threads load tree_branch elements and sum them
        // For factor=32, tree_branch 4:
        //   level 0: stride=32, active=8, offsets=8,16,24 -> combine for i in 0..8
        //   level 1: stride=8, active=2, offsets=2,4,6 -> combine for i in 0..2
        //   level 2: stride=2 < tree_branch=4, exit first loop
        //   Then binary reduction: stride=2 -> 1
        let mut stride = factor;
        while stride > 1 {
            let use_tree_branch = stride >= tree_branch;
            let active_threads = if use_tree_branch { stride / tree_branch } else { stride / 2 };
            let limit_const = self.insert_before(acc_load_id, Op::Const(Constant::idx(active_threads)));
            let condition = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: limit_const, bop: BOp::Cmplt });
            self.insert_before(acc_load_id, Op::Loop { len: condition });

            let branch = if use_tree_branch { tree_branch } else { 2 };
            let mut sum_x = None;
            for i in 1..branch {
                let offset = i * active_threads;
                let offset_const = self.insert_before(acc_load_id, Op::Const(Constant::idx(offset)));
                let offset_idx = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: offset_const, bop: BOp::Add });
                let local_load = self.insert_before(acc_load_id, Op::Load { src: loc_acc, src_idx: offset_idx });
                if let Some(prev_sum) = sum_x {
                    sum_x = Some(self.insert_before(acc_load_id, Op::Binary { x: prev_sum, y: local_load, bop }));
                } else {
                    let current_val = self.insert_before(acc_load_id, Op::Load { src: loc_acc, src_idx: lidx });
                    sum_x = Some(self.insert_before(acc_load_id, Op::Binary { x: current_val, y: local_load, bop }));
                }
            }
            let bop_id = sum_x.unwrap();
            self.insert_before(acc_load_id, Op::Store { dst: loc_acc, dst_idx: lidx, src: bop_id });

            self.insert_before(acc_load_id, Op::EndLoop);
            self.insert_before(acc_load_id, Op::Barrier);

            stride = active_threads;
        }

        // Load final result from local[0] to register (only thread 0)
        let condition = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: const_zero, bop: BOp::Eq });
        self.insert_before(acc_load_id, Op::Loop { len: condition });
        let final_val = self.insert_before(acc_load_id, Op::Load { src: loc_acc, src_idx: const_zero });
        self.insert_before(acc_load_id, Op::Store { dst: reg_acc, dst_idx: const_zero, src: final_val });
        self.insert_after(self.tail, Op::EndLoop);

        self.verify();
    }
}

// ===========================================================================
// The int8 vec-dot form
// ===========================================================================
//
// The PQ2_0 decode projection does not reduce a bare accumulator: its K loop
// is a loop over 32-element groups, each group a `q8_1` chunk whose integer
// dot (`dp4a`) is scaled once by `f16(weight scale) * f16(act scale)`. The
// general split above cannot express that, and it also refuses a kernel with
// a nested loop. `vec_dot_split` is the shape the reference decode kernel
// runs (`mul_mat_vec_q<PQ2_0, 1>`, `ggml/src/ggml-cuda/mmvq.cu`):
//
// - the group loop is split over `threads` lanes: thread `tid` takes group
//   `tid + threads*i`, which for `threads = 128` is the reference's
//   `tid % 4` chunk of super-block `tid / 4 + 32*i`;
// - the per-element body is replaced with four `u16` code words expanded by
//   `prmt` into eight `int8` quads and eight `dp4a` against the chunk's eight
//   `u32` activation words;
// - the partials reduce in two levels: the warps other than 0 stage their
//   per-lane partial in shared memory, warp 0 folds them into its own and
//   reduces over the 32 lanes with `shfl.sync.bfly`.
//
// The integer dot is exact, so the rewrite changes no value: the only
// difference from the naive form is the f32 fold order across the lanes.

/// Elements per `q8_1` group (the chunk width).
const VD_ELEMS: i64 = 32;
/// Threads that share one 128-weight super-block (`QK_PQ2_0 / 32`).
const VD_QI: i64 = 4;
/// `u16` words per PQ2_0 super-block: the f16 scale, then 16 code words.
const VD_WORDS: i64 = 17;
/// The code-to-value table: byte `i` of `0x020100FF` is `i - 1`.
const VD_TABLE: u32 = 0x0201_00FF;
/// Selector mask: bit 3 of a `prmt` selector is sign-replicate.
const VD_SEL_MASK: u16 = 0x7777;
/// The even/odd output selectors of `__byte_perm(qe, qo, ...)`.
const VD_SEL_EVEN: u32 = 0x5140;
/// The odd half of the pair.
const VD_SEL_ODD: u32 = 0x7362;

/// What the rewriter needs from the kernel it matched.
#[derive(Debug, Clone, Copy)]
struct VecDotShape {
    /// The group loop and its closing `EndLoop`.
    loop_id: OpId,
    /// The row store after the loop (the rewrite rebuilds it).
    end_store: OpId,
    /// Groups along K (each [`VD_ELEMS`] elements).
    groups: i64,
    /// Activation payload words (`U32`, four `int8` each).
    a8: OpId,
    /// Activation scale words (`U32`, `half2(d, sum)` per group).
    ds: OpId,
    /// PQ2_0 super-blocks as `u16` words.
    w: OpId,
    /// The output row vector (`F32`).
    out: OpId,
    /// The output row index.
    row: OpId,
    /// The row's f32 register accumulator.
    acc: OpId,
}

impl Kernel {
    /// Match the int8 vec-dot reduction skeleton: a 32-element group loop
    /// with an integer chunk accumulator and a scaled f32 row accumulator.
    ///
    /// Returns `None` for every other shape, so a kernel that is not this
    /// form is offered no config (a miss, never a wrong rewrite).
    fn vec_dot_shape(&self, loop_id: OpId) -> Option<VecDotShape> {
        let Op::Loop { len } = self.ops[loop_id].op else {
            return None;
        };
        let groups = self.resolve_const(len).and_then(Constant::as_dim)?;
        // The group loop must cover whole super-blocks (four chunks each).
        if groups < 2 || groups % VD_QI != 0 {
            return None;
        }

        let mut depth = 1i32;
        let mut inner = OpId::NULL;
        let mut inner_end = OpId::NULL;
        let mut end_loop = OpId::NULL;
        let mut w_code = OpId::NULL;
        let mut a_payload = OpId::NULL;
        let mut w_scale = OpId::NULL;
        let mut ds_load = OpId::NULL;
        let mut acc = OpId::NULL;
        let mut id = self.next_op(loop_id);
        while !id.is_null() {
            match self.ops[id].op {
                Op::Loop { .. } => {
                    depth += 1;
                    if depth == 2 {
                        inner = id;
                    }
                }
                Op::EndLoop => {
                    depth -= 1;
                    if depth == 0 {
                        end_loop = id;
                        break;
                    }
                    if depth == 1 {
                        inner_end = id;
                    }
                }
                Op::Load { src, .. } => {
                    if let Op::Param { dtype, kind, .. } = self.ops[src].op
                        && kind != ParamKind::GlobalMut
                    {
                        match (dtype, depth) {
                            (DType::U16, 2) if w_code.is_null() => w_code = id,
                            (DType::U32, 2) if a_payload.is_null() => a_payload = id,
                            (DType::U16, 1) if w_scale.is_null() => w_scale = id,
                            (DType::U32, 1) if ds_load.is_null() => ds_load = id,
                            _ => {}
                        }
                    }
                }
                Op::Store { dst, .. } => {
                    if acc.is_null()
                        && let Op::Storage { dtype: DType::F32, scope: MemScope::Register, len: 1, .. } = self.ops[dst].op
                    {
                        acc = dst;
                    }
                }
                _ => {}
            }
            id = self.next_op(id);
        }
        if end_loop.is_null() || inner.is_null() || inner_end.is_null() {
            return None;
        }
        // The inner loop is the 32-element chunk.
        let Op::Loop { len } = self.ops[inner].op else {
            return None;
        };
        if self.resolve_const(len).and_then(Constant::as_dim)? != VD_ELEMS {
            return None;
        }
        for x in [w_code, a_payload, w_scale, ds_load, acc] {
            if x.is_null() {
                return None;
            }
        }
        let Op::Load { src: a8, .. } = self.ops[a_payload].op else {
            return None;
        };
        let Op::Load { src: w, .. } = self.ops[w_code].op else {
            return None;
        };
        let Op::Load { src: ds, .. } = self.ops[ds_load].op else {
            return None;
        };
        let _ = w_scale;
        // The row index: the weight code word's index cone names a group range.
        let Op::Load { src_idx, .. } = self.ops[w_code].op else {
            return None;
        };
        let row = self.group_index_in(src_idx)?;
        // The output: the store after the loop to a mutable f32 param, and it
        // must be the kernel's last op (the rewrite rebuilds the tail).
        let mut out = OpId::NULL;
        let mut out_store = OpId::NULL;
        let mut id = self.next_op(end_loop);
        while !id.is_null() {
            if out.is_null()
                && let Op::Store { dst, dst_idx, .. } = self.ops[id].op
                && let Op::Param { dtype: DType::F32, kind: ParamKind::GlobalMut, .. } = self.ops[dst].op
            {
                if dst_idx != row || !self.next_op(id).is_null() {
                    return None;
                }
                out = dst;
                out_store = id;
            }
            id = self.next_op(id);
        }
        if out.is_null() {
            return None;
        }
        Some(VecDotShape { loop_id, end_store: out_store, groups, a8, ds, w, out, row, acc })
    }

    /// The group range an index expression is built from, if any.
    fn group_index_in(&self, x: OpId) -> Option<OpId> {
        let mut stack = vec![x];
        let mut seen: Vec<OpId> = Vec::new();
        while let Some(id) = stack.pop() {
            if id.is_null() || seen.contains(&id) {
                continue;
            }
            seen.push(id);
            match self.ops[id].op {
                Op::Range { kind: RangeKind::Group(_), .. } => return Some(id),
                ref op => stack.extend(op.parameters().filter(|p| !p.is_null())),
            }
        }
        None
    }

    /// Rewrite the int8 vec-dot form into the split fast form: `threads`
    /// lanes (`32 * warps`) over the group loop, the `prmt`/`dp4a` chunk
    /// body, and the two-level warp/shared reduction.
    pub(crate) fn vec_dot_split(&mut self, loop_id: OpId, threads: u32) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("vec_dot_split");
        let shape = self.vec_dot_shape(loop_id).expect("vec_dot_split: kernel is not the int8 vec-dot form");
        assert!(threads % 32 == 0, "vec_dot_split: a split is a whole number of warps, got {threads}");
        assert_eq!(shape.loop_id, loop_id);

        let warps = threads / 32;
        let groups = shape.groups;
        let blk_count = groups / VD_QI;
        let trips = (groups + i64::from(threads) - 1) / i64::from(threads);

        // Drop the naive region: the group loop and the row store that reads
        // its accumulator. What is left in front is the accumulator's own
        // zero-init store, so the rewrite can be appended at the tail and
        // every op still follows its operands.
        let mut id = loop_id;
        loop {
            let next = self.next_op(id);
            let last = id == shape.end_store;
            self.remove_op(id);
            if last {
                break;
            }
            id = next;
        }

        // ---- the lanes ----
        let lane = self.push_back(Op::Range { axis: 0, kind: RangeKind::Local(32) });
        let tid = if warps > 1 {
            let warp = self.push_back(Op::Range { axis: 1, kind: RangeKind::Local(warps) });
            let w32 = self.const_idx(i64::from(32));
            let wrow = self.push_back(Op::Binary { x: warp, y: w32, bop: BOp::Mul });
            self.push_back(Op::Binary { x: wrow, y: lane, bop: BOp::Add })
        } else {
            lane
        };

        // ---- constants ----
        let c_qi = self.const_idx(VD_QI);
        let c_words = self.const_idx(VD_WORDS);
        let c_row_words = self.const_idx(blk_count * VD_WORDS);
        let c_groups = self.const_idx(groups);
        let c_steps = self.const_idx(i64::from(threads));
        let c_one = self.const_idx(1);
        let c_four = self.const_idx(4);
        let c_eight = self.const_idx(8);
        let c_zero_i = self.const_idx(0);
        let c32 = self.const_idx(i64::from(32));
        let sel_mask = self.const_val(VD_SEL_MASK);
        let table = self.const_val(VD_TABLE);
        let sel_even = self.const_val(VD_SEL_EVEN);
        let sel_odd = self.const_val(VD_SEL_ODD);
        let mask16 = self.const_val(0xffffu32);
        let two_u16 = self.const_val(2u16);
        let zero_u32 = self.const_val(0u32);

        // The shared staging tile: one f32 per lane of the warps after the
        // first.
        let sh_len = i64::from(32 * warps);
        let sh = self.push_back(Op::Storage { dtype: DType::F32, scope: MemScope::Local, len: sh_len, id: 0 });

        // ---- the split group loop ----
        // A loop op is its own index value.
        let c_trips = self.const_idx(trips);
        let it = self.push_back(Op::Loop { len: c_trips });
        let step = self.push_back(Op::Binary { x: it, y: c_steps, bop: BOp::Mul });
        let g = self.push_back(Op::Binary { x: step, y: tid, bop: BOp::Add });
        let in_range = self.push_back(Op::Binary { x: g, y: c_groups, bop: BOp::Cmplt });
        self.push_back(Op::Loop { len: in_range });
        {
            let blk = self.push_back(Op::Binary { x: g, y: c_qi, bop: BOp::Div });
            let chunk = self.push_back(Op::Binary { x: g, y: c_qi, bop: BOp::Mod });
            // The row's super-block base plus this group's block.
            let row_words = self.push_back(Op::Binary { x: shape.row, y: c_row_words, bop: BOp::Mul });
            let blk_words = self.push_back(Op::Binary { x: blk, y: c_words, bop: BOp::Mul });
            let base17 = self.push_back(Op::Binary { x: row_words, y: blk_words, bop: BOp::Add });
            // The super-block's f16 scale.
            let wd16 = self.push_back(Op::Load { src: shape.w, src_idx: base17 });
            let wd16 = self.push_back(Op::Bitcast { x: wd16, dtype: DType::F16 });
            let wd = self.push_back(Op::Cast { x: wd16, dtype: DType::F32 });
            // The group's f16 activation scale: the low half of its `ds` word.
            let dsw = self.push_back(Op::Load { src: shape.ds, src_idx: g });
            let ds_lo = self.push_back(Op::Binary { x: dsw, y: mask16, bop: BOp::BitAnd });
            let ds_u16 = self.push_back(Op::Cast { x: ds_lo, dtype: DType::U16 });
            let ds_f16 = self.push_back(Op::Bitcast { x: ds_u16, dtype: DType::F16 });
            let ad = self.push_back(Op::Cast { x: ds_f16, dtype: DType::F32 });
            let scale = self.push_back(Op::Binary { x: wd, y: ad, bop: BOp::Mul });

            // Eight `dp4a`: four code words, each expanded into two `int8`
            // quads by `prmt`, against the chunk's eight activation words.
            let mut sumi = zero_u32;
            for j in 0..4i64 {
                let cj = self.const_idx(j);
                let c2j = self.const_idx(2 * j);
                let word = self.push_back(Op::Binary { x: base17, y: c_one, bop: BOp::Add });
                let chunk4 = self.push_back(Op::Binary { x: chunk, y: c_four, bop: BOp::Mul });
                let word = self.push_back(Op::Binary { x: word, y: chunk4, bop: BOp::Add });
                let word = self.push_back(Op::Binary { x: word, y: cj, bop: BOp::Add });
                let q = self.push_back(Op::Load { src: shape.w, src_idx: word });
                let qe_m = self.push_back(Op::Binary { x: q, y: sel_mask, bop: BOp::BitAnd });
                let qe_sel = self.push_back(Op::Cast { x: qe_m, dtype: DType::U32 });
                let q_sh = self.push_back(Op::Binary { x: q, y: two_u16, bop: BOp::BitShiftRight });
                let qo_m = self.push_back(Op::Binary { x: q_sh, y: sel_mask, bop: BOp::BitAnd });
                let qo_sel = self.push_back(Op::Cast { x: qo_m, dtype: DType::U32 });
                let qe = self.ptx(PTXOp::Prmt { a: table, b: table, c: qe_sel });
                let qo = self.ptx(PTXOp::Prmt { a: table, b: table, c: qo_sel });
                let qx = self.ptx(PTXOp::Prmt { a: qe, b: qo, c: sel_even });
                let qy = self.ptx(PTXOp::Prmt { a: qe, b: qo, c: sel_odd });
                let aw = self.push_back(Op::Binary { x: g, y: c_eight, bop: BOp::Mul });
                let aw = self.push_back(Op::Binary { x: aw, y: c2j, bop: BOp::Add });
                let u = self.push_back(Op::Load { src: shape.a8, src_idx: aw });
                let aw1 = self.push_back(Op::Binary { x: aw, y: c_one, bop: BOp::Add });
                let v = self.push_back(Op::Load { src: shape.a8, src_idx: aw1 });
                sumi = self.ptx(PTXOp::Dp4a { a: u, b: qx, c: sumi });
                sumi = self.ptx(PTXOp::Dp4a { a: v, b: qy, c: sumi });
            }
            let si = self.push_back(Op::Bitcast { x: sumi, dtype: DType::I32 });
            let si = self.push_back(Op::Cast { x: si, dtype: DType::F32 });
            let part = self.push_back(Op::Binary { x: scale, y: si, bop: BOp::Mul });
            let prev = self.push_back(Op::Load { src: shape.acc, src_idx: c_zero_i });
            let total = self.push_back(Op::Binary { x: prev, y: part, bop: BOp::Add });
            self.push_back(Op::Store { dst: shape.acc, dst_idx: c_zero_i, src: total });
        }
        self.push_back(Op::EndLoop);
        self.push_back(Op::EndLoop);

        // ---- two-level reduce: cross-warp in shared, then warp shuffle ----
        let warp_on = self.push_back(Op::Binary { x: tid, y: c32, bop: BOp::Cmpge });
        self.push_back(Op::Loop { len: warp_on });
        {
            let rel = self.push_back(Op::Binary { x: tid, y: c32, bop: BOp::Sub });
            let acc_v = self.push_back(Op::Load { src: shape.acc, src_idx: c_zero_i });
            self.push_back(Op::Store { dst: sh, dst_idx: rel, src: acc_v });
        }
        self.push_back(Op::EndLoop);
        self.push_back(Op::Barrier);
        // Warp 0 folds the other warps' per-lane partials, then reduces over
        // the 32 lanes and writes the row.
        let warp0 = self.push_back(Op::Binary { x: tid, y: c32, bop: BOp::Cmplt });
        self.push_back(Op::Loop { len: warp0 });
        let mut s = self.push_back(Op::Load { src: shape.acc, src_idx: c_zero_i });
        for w in 1..warps {
            // Warp `w` staged its partial at `(w - 1) * 32 + lane`.
            let off = self.const_idx(i64::from(w - 1) * 32);
            let at = self.push_back(Op::Binary { x: off, y: lane, bop: BOp::Add });
            let v = self.push_back(Op::Load { src: sh, src_idx: at });
            s = self.push_back(Op::Binary { x: s, y: v, bop: BOp::Add });
        }
        // The warp butterfly: five `shfl.sync.bfly` exchanges of one f32 word.
        for m in [16u32, 8, 4, 2, 1] {
            let mask = self.const_val(m);
            let shf = self.ptx(PTXOp::ShflXor { value: s, mask });
            s = self.push_back(Op::Binary { x: s, y: shf, bop: BOp::Add });
        }
        let lane0 = self.push_back(Op::Binary { x: lane, y: c_zero_i, bop: BOp::Eq });
        self.push_back(Op::Loop { len: lane0 });
        self.push_back(Op::Store { dst: shape.out, dst_idx: shape.row, src: s });
        self.push_back(Op::EndLoop);
        self.push_back(Op::EndLoop);

        self.verify();
    }

    /// A PTX instruction appended at the tail.
    fn ptx(&mut self, op: PTXOp) -> OpId {
        self.push_back(Op::PTX(Box::new(op)))
    }
}
