// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
// Tenstorrent tiling.
//
// `tt_tile` is the last zyx-side pass on the Tenstorrent path: it turns the
// search's loops and group ranges into the 32-wide tile IR. Everything after
// this point — the movers, circular-buffer hand-offs, `Dst` allocation, the
// RISC-V and datapath encoders — belongs to `libskamandros.so`, which compiles
// the tiled program that `codegen::skamandros` writes out.

use crate::Set;
use crate::dtype::Constant;
use crate::kernel::{BOp, Kernel, Op, OpId, RangeKind, SplitMode};

impl Kernel {
    /// Tenstorrent tiling: pad to multiples of 32 and unroll loops
    /// and group ranges to multiples of 32. No special tiling or
    /// vectorization passes: the 32-wide `Unroll` lanes plus composed
    /// indices are the tile IR the backend matchers read.
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        self.default_epilogue();
        self.default_epilogue();
        // Normalize the loop tiling the search left to the one shape the
        // tile IR has: `Loop(k_tiles) + Unroll(32)`. Three input shapes
        // reach here.
        //
        // - A loop in **element** units (the unsplit reduce): pad to 32 and
        //   split. That is where the lane axis and the `Reduce` come from.
        // - A loop the search **strip-mined** (`SplitMode::Loop`:
        //   `factor*outer + inner`): fold the pair back into one loop of
        //   `outer*inner` trips, then pad + split. Padding the outer alone
        //   counts the inner trips again: softmax's 8-tile reduce became
        //   32 iterations and a 4 MB reader block.
        // - A loop the search already **split into lanes**
        //   (`SplitMode::Unroll`): its trip count is tiles, not elements,
        //   so it is left alone (same over-read if padded).
        //
        // Folding validates the whole pair before mutating: the inner var's
        // only use is the composition split leaves right after the inner
        // header, the outer var is used by nothing inside the inner body, and
        // no side effect sits in the outer body between the two loops (it
        // would run once per flat iteration instead of once per outer one).
        for _ in 0..64 {
            let mut inner_ops: Vec<OpId> = Vec::new();
            for (op, x) in self.iter_unordered() {
                if matches!(x, Op::Loop { .. }) {
                    inner_ops.push(op);
                }
            }
            let mut folded = false;
            for inner in inner_ops {
                let Some(outer) = self.tt_enclosing_loop(inner) else {
                    continue;
                };
                let composed = self.next_op(inner);
                if composed.is_null() {
                    continue;
                }
                let Op::Binary { x, y, bop: BOp::Add } = self.ops[composed].op else {
                    continue;
                };
                let acc = if y == inner {
                    x
                } else if x == inner {
                    y
                } else {
                    continue;
                };
                if !self.depends_on(acc, outer, &mut Set::default()) {
                    continue;
                }
                let (Op::Loop { len: outer_len }, Op::Loop { len: inner_len }) =
                    (self.ops[outer].op.clone(), self.ops[inner].op.clone())
                else {
                    continue;
                };
                let (Some(ol), Some(il)) = (
                    self.resolve_const(outer_len).and_then(Constant::as_dim),
                    self.resolve_const(inner_len).and_then(Constant::as_dim),
                ) else {
                    continue;
                };
                // The fold rewrites the composition's value to the flat loop
                // variable, which enumerates `outer * il + inner`. The
                // substitution is exact only when the accumulator side
                // advances by exactly `il` per outer trip: in
                // `acc + inner`, `acc` must have outer coefficient `il`.
                // `depends_on` alone is not enough — a strip-mined index that
                // also carries a base (`row * cols + outer * 32`, what a
                // fused reduce over `exp(x - max)` produces) would lose that
                // base. The invariant part of `acc` is kept below.
                let Some(strides) = self.get_strides(acc) else {
                    continue;
                };
                let Some(&(_, scale)) = strides.get(&outer) else {
                    continue;
                };
                if scale != il {
                    continue;
                }
                let inner_end = self.get_last_dim_op(inner);
                let outer_end = self.get_last_dim_op(outer);
                // The inner var belongs to the composition alone.
                let mut ok = true;
                for (op, x) in self.iter_unordered() {
                    if op != composed && x.parameters().any(|p| p == inner) {
                        ok = false;
                        break;
                    }
                }
                // A value is *tainted* when it reaches the outer var without
                // going through the composition. After folding the outer var
                // enumerates the flat trips, so a tainted value read inside
                // the inner body (or after it) would see the wrong index. The
                // composition itself bridges the two loop names: its consumers
                // are downstream of the flat index and stay correct.
                let mut tainted: Set<OpId> = Set::default();
                let mut walk = self.head;
                while !walk.is_null() {
                    if walk != composed
                        && self.ops[walk].op.parameters().any(|p| p == outer || (p != composed && tainted.contains(&p)))
                    {
                        tainted.insert(walk);
                    }
                    walk = self.next_op(walk);
                }
                // Between the two loops: the composition's own math, pure.
                let mut scan = self.next_op(outer);
                while ok && !scan.is_null() && scan != inner {
                    if matches!(self.ops[scan].op, Op::Store { .. } | Op::Copy { .. }) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                // The inner body may write its accumulator, but no tainted
                // value may reach it.
                let mut scan = self.next_op(inner);
                while ok && !scan.is_null() && scan != inner_end {
                    if scan != composed && tainted.contains(&scan) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                // After the inner loop, still in the outer body: no side
                // effects (they would run per flat trip), no tainted value.
                let mut scan = self.next_op(inner_end);
                while ok && !scan.is_null() && scan != outer_end {
                    if matches!(self.ops[scan].op, Op::Store { .. } | Op::Copy { .. }) || tainted.contains(&scan) {
                        ok = false;
                    }
                    scan = self.next_op(scan);
                }
                if !ok {
                    continue;
                }
                let flat = self.insert_const_idx_before(outer, ol * il);
                self.ops[outer].op = Op::Loop { len: flat };
                // `composed` is `acc + inner`. With the outer var now
                // enumerating `flat = outer*il + inner`, the same value is
                // `(acc - flat*il) + flat`: `acc - outer*il` is the part of
                // the index that does not advance with the outer trip (a row
                // base, a zero), and `flat` replaces the strip-mined pair.
                let il_c = self.insert_const_idx_before(composed, il);
                let scaled = self.insert_before(composed, Op::Binary { x: outer, y: il_c, bop: BOp::Mul });
                let base = self.insert_before(composed, Op::Binary { x: acc, y: scaled, bop: BOp::Sub });
                let folded_idx = self.insert_before(composed, Op::Binary { x: base, y: outer, bop: BOp::Add });
                self.remap(composed, folded_idx);
                self.remove_op(composed);
                self.remove_op(inner_end);
                self.remove_op(inner);
                folded = true;
                break;
            }
            if !folded {
                break;
            }
        }
        // Gather loops, pad to 32, split unroll-32 FIRST: split's
        // loop arm allows exactly one acc store per body, so it must
        // run before coarsening replicates the body. pad_to pads
        // lengths only on TT (no guards); already-multiples are
        // untouched. Non-reduce loops are a no-op inside (no acc
        // found); anything else keeps its shape for later cuts.
        let mut pad_loops: Vec<(OpId, bool)> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if !matches!(x, Op::Loop { .. }) {
                continue;
            }
            // `split(Loop, Unroll, 32)` leaves `loop*32 + lane`: the loop
            // advances one 32-lane chunk (one tile) per trip, so its length
            // is tiles already.
            let mut tiled = false;
            let end = self.get_last_dim_op(op);
            let mut scan = self.next_op(op);
            while !scan.is_null() && scan != end && !tiled {
                let idx = match self.ops[scan].op {
                    Op::Load { src_idx, .. } => src_idx,
                    Op::Store { dst_idx, .. } => dst_idx,
                    Op::Copy { src_idx, .. } => src_idx,
                    _ => OpId::NULL,
                };
                if !idx.is_null()
                    && let Some(strides) = self.get_strides(idx)
                    && strides.get(&op).is_some_and(|&(_, s)| s == 32)
                    && strides
                        .iter()
                        .any(|(v, &(_, s))| *v != op && s == 1 && matches!(self.ops[*v].op, Op::Unroll { len: 32, .. }))
                {
                    tiled = true;
                }
                scan = self.next_op(scan);
            }
            pad_loops.push((op, tiled));
        }
        for (l, tiled) in pad_loops {
            if tiled {
                continue;
            }
            self.pad_to(l, 32);
            let len = match self.ops[l].op {
                Op::Loop { len, .. } => len,
                _ => unreachable!("pad_to preserves the Loop variant"),
            };
            // resolve_const evaluates const-expr lengths on demand, so
            // freshly padded (unfolded) round-ups resolve directly.
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(l, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();
        // Gather group ranges, pad to 32, coarsen LAST: duplicate
        // the vectorized body, one lane per row. Anything else
        // (symbolic, indivisible) is skipped, never fatal. The device
        // grid is 2D: one group range is pre-split into two nested
        // groups below, two proceed as-is, more than two is a later
        // increment.
        //
        // The contract (NOTES, "Tiling: the search explores, `tt_tile`
        // completes"): the search is free to tile, `opt_split` never splits
        // past what the device can hold, and this pass **fills the remaining
        // gaps** — it adds the tile axes the search did not, and touches
        // nothing that is already tiled. An access's index rank is the number
        // of tile axes already present.
        let elementwise = !self.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. }));
        let axes = self.device_info().tile.len();
        // `Op::Reduce` is not a "reduce kernel": `linearize` turns a reduce
        // into a `Loop` with a scalar accumulator, and a `Reduce` only exists
        // here once `split` has unrolled such a loop into lanes. That case
        // keeps its extra axes — the rank-3 cube is deliberate, and the
        // backend rewrites it — but only when the reduce loop is gone. With a
        // reduce loop still present the template downstream is one tile per
        // iteration, i.e. the reduce lane plus one row axis. Splitting a group
        // on top of that makes a rank-3 access whose split group is a *grid*
        // axis padded into lanes; that padded lane reads past the tensor. So
        // the group split gets the same rank budget as the elementwise case:
        // `axes` lanes while a loop remains, `axes + 1` when the loop was fully
        // split into lanes.
        let rank = self.iter_unordered().fold(0, |rank, (_, op)| {
            let idx = match *op {
                Op::Load { src_idx, .. } => src_idx,
                Op::Store { dst_idx, .. } => dst_idx,
                Op::Copy { src_idx, .. } => src_idx,
                _ => return rank,
            };
            if idx.is_null() {
                rank
            } else {
                rank.max(self.shape(idx).len())
            }
        });
        let mut pad_groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                pad_groups.push(op);
            }
        }
        if pad_groups.len() > 2 {
            todo!("tt_tile: more than two group ranges");
        }
        // A group already counted in **tiles** is the one the search vectorized
        // along: its term is `32 * g + lane`, with `lane` an `Unroll` of 32.
        // That group is the dimension's grid axis and must be left as it is
        // (padding 8 column tiles to 32 would turn them into a lane axis and
        // leave the row group an element grid: `[10, 256]` unary, grid
        // `[10, 1]`, one 4 KB page per row). The other group counts
        // **elements**; it is the one padded to 32 and split into the axis
        // being added. A bare `g * 32` is not the test — 32 is also a legal
        // row stride (`[10, 32]`'s rows are `g * 32`), and what marks a
        // vectorized dimension is the lane **added** to the scaled group.
        let mut chunked: Vec<OpId> = Vec::new();
        if elementwise {
            for (_, op) in self.iter_unordered() {
                let Op::Binary { x, y, bop: BOp::Add } = *op else { continue };
                for (scaled, lane) in [(x, y), (y, x)] {
                    if !matches!(self.ops[lane].op, Op::Unroll { .. }) {
                        continue;
                    }
                    let Op::Binary { x: m, y: n, bop: BOp::Mul } = self.ops[scaled].op else {
                        continue;
                    };
                    for (g, c) in [(m, n), (n, m)] {
                        if matches!(self.ops[g].op, Op::Range { kind: RangeKind::Group(_), .. })
                            && self.resolve_const(c).and_then(Constant::as_dim) == Some(32)
                        {
                            chunked.push(g);
                        }
                    }
                }
            }
        }
        pad_groups.sort_by_key(|g| match self.ops[*g].op {
            Op::Range { axis, .. } => axis,
            _ => unreachable!("pad_groups holds group ranges"),
        });
        let mut need = if elementwise {
            axes.saturating_sub(rank)
        } else if self.iter_unordered().any(|(_, op)| matches!(op, Op::Loop { .. })) {
            axes.saturating_sub(rank)
        } else {
            (axes + 1).saturating_sub(rank)
        };
        // An access that already carries a lane axis and a group is a 2-D
        // access: the group *is* the grid axis, so no axis is missing. Padding
        // the group into lanes would turn every row into a lane of one grid
        // point and mask it off (the gather's `idx[s]`), which is not a tiling.
        if rank >= 1 && self.iter_unordered().any(|(_, op)| matches!(op, Op::Range { kind: RangeKind::Group(_), .. })) {
            need = 0;
        }
        for g in pad_groups {
            if need == 0 {
                break;
            }
            if chunked.contains(&g) {
                continue;
            }
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => len,
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(g, SplitMode::Unroll, 32);
                need -= 1;
            }
        }

        // The launch grid is the device's Tensix grid: a group axis longer than
        // `max_global_work_dims[axis]` cannot be dispatched, so it is split
        // into an outer group the device can hold and a loop over the rest
        // (`split`'s `outer * factor + inner`, `factor` the smallest divisor
        // of the length that leaves the outer part within the limit). The
        // loop wraps the whole kernel body, so every iteration is one of the
        // group's points.
        let max_grid = self.device_info().max_global_work_dims.clone();
        let mut oversized: Vec<(OpId, i64)> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if let Op::Range { axis, kind: RangeKind::Group(len) } = *x {
                let Some(l) = self.resolve_const(len).and_then(Constant::as_dim) else {
                    continue;
                };
                let Some(&max) = max_grid.get(axis as usize) else {
                    continue;
                };
                if l > max {
                    oversized.push((op, max));
                }
            }
        }
        for (group, max) in oversized {
            let Op::Range { kind: RangeKind::Group(len), .. } = self.ops[group].op else {
                unreachable!("tt_tile: collected op is a group range");
            };
            let Some(l) = self.resolve_const(len).and_then(Constant::as_dim) else {
                continue;
            };
            let min_factor = (l + max - 1) / max;
            let Some(factor) = (min_factor..=l).find(|d| l % d == 0) else {
                todo!("tt_tile: group range {l} has no divisor leaving ≤ {max}")
            };
            let Ok(factor) = u16::try_from(factor) else {
                todo!("tt_tile: group split factor does not fit u16")
            };
            self.split(group, SplitMode::Loop, factor);
        }

        self.default_epilogue();
        self.default_epilogue();
    }

    fn tt_enclosing_loop(&self, at: OpId) -> Option<OpId> {
        let mut stack: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() || id == at {
                break;
            }
            match self.ops[id].op {
                Op::Loop { .. } => stack.push(id),
                Op::EndLoop => {
                    stack.pop();
                }
                _ => {}
            }
            id = self.next_op(id);
        }
        stack.last().copied()
    }
}
