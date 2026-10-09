// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tiling pass: scalar row-major traffic becomes tile traffic.
//! (Under construction, one step at a time.)

use super::{BOp, Kernel, MemLayout, Op, OpId, ParamKind, RangeKind};
use crate::shape::Dim;

impl Kernel {
    /// Tenstorrent tiling pass
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let tilew: Dim = 32;
        let mut axes: [(OpId, Dim); 2] = [(OpId::NULL, 0); 2];
        for (op, x) in self.iter_unordered() {
            if let Op::Loop { .. } = x {
                return;
            }
            if let &Op::Range { axis, kind: RangeKind::Group(len) } = x {
                if axis as usize >= axes.len() {
                    return;
                }
                axes[axis as usize] = (op, self.resolve_const(len).and_then(|c| c.as_dim()).unwrap());
            }
        }
        if axes[0].0.is_null() || axes[1].0.is_null() {
            return;
        }
        for (_, len) in axes.iter() {
            if len % tilew != 0 {
                return;
            }
        }

        // TODO put here the get_strides checks for all geps
        // Every traffic side must be a scalar DRAM GEP indexed densely
        // row-major over exactly the two groups: group-0 stride is len1,
        // group-1 stride is 1, const offset is 0. The stride values feed
        // the tile-offset formula below; they are not a second opinion
        // on get_strides.
        let mut geps: Vec<OpId> = Vec::new();
        for (_, x) in self.iter_unordered() {
            match x {
                Op::Load { src } => geps.push(*src),
                Op::Store { dst, .. } => geps.push(*dst),
                Op::Copy { src, dst } => {
                    geps.push(*src);
                    geps.push(*dst);
                }
                _ => {}
            }
        }
        for gep in geps.iter() {
            let Op::GEP { x: base, index, layout } = self.ops[*gep].op else {
                return;
            };
            if layout != MemLayout::Scalar {
                return;
            }
            if !matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                return;
            }
            let strides = self.get_strides(index);
            if strides.get(&OpId::NULL).is_some_and(|(_, off)| *off != 0) {
                return;
            }
            let mut n = 0;
            for (k, (_, s)) in strides.iter() {
                if *k == OpId::NULL {
                    continue;
                }
                if (*k == axes[0].0 && *s == axes[1].1) || (*k == axes[1].0 && *s == 1) {
                    n += 1;
                } else {
                    return;
                }
            }
            if n != 2 {
                return;
            }
        }
        // Rewrite each GEP to its tilized tile offset
        // (g0 * tl1 + g1) * 1024 and flip it to 32x32 tiles. Group op
        // ids still address the groups (division rewrote them in place).
        let tl1 = axes[1].1 / tilew;
        for gep in geps.iter() {
            let Op::GEP { x: base, .. } = self.ops[*gep].op else { return };
            let ctl1 = self.insert_const_idx_before(*gep, tl1);
            let tile_id = self.insert_before(*gep, Op::Binary { x: axes[0].0, y: ctl1, bop: BOp::Mul });
            let tile_id = self.insert_before(*gep, Op::Binary { x: tile_id, y: axes[1].0, bop: BOp::Add });
            let c1024 = self.insert_const_idx_before(*gep, tilew * tilew);
            let offset = self.insert_before(*gep, Op::Binary { x: tile_id, y: c1024, bop: BOp::Mul });
            self.ops[*gep].op = Op::GEP { x: base, index: offset, layout: MemLayout::Tile { x: 32, y: 32, stride: 32 } };
        }

        for (op, ..) in axes.iter() {
            let Op::Range { axis, kind: RangeKind::Group(len_op) } = self.ops[*op].op else {
                return;
            };
            let thirtytwo = self.insert_const_idx_before(*op, tilew);
            let short = self.insert_before(*op, Op::Binary { x: len_op, y: thirtytwo, bop: BOp::Div });
            self.ops[*op].op = Op::Range { axis, kind: RangeKind::Group(short) };
        }
        self.verify();
    }
}

#[cfg(test)]
mod tests {
    use super::Kernel;
    use crate::dtype::DType;
    use crate::kernel::{Dev, MemLayout, Op, RangeKind};

    #[test]
    fn tt_tile_2d() {
        let mut k = Kernel::new(Dev::TT(0));
        let a = k.param(DType::BF16);
        let o = k.param_mut(DType::BF16);
        let rows = k.const_idx(32);
        let cols = k.const_idx(256);
        let gr = k.group_range(0, rows);
        let gc = k.group_range(1, cols);
        let stride = k.mul(gr, cols);
        let idx = k.add(gc, stride);
        let v = k.load(a, idx);
        let n = k.neg(v);
        k.store(o, n, idx);
        k.tt_tile();
        for (range, expected) in [(gr, 1), (gc, 8)] {
            let Op::Range { kind: RangeKind::Group(len), .. } = k.at(range) else {
                panic!("tt_tile_2d: group is not a Group range");
            };
            assert_eq!(k.resolve_const(*len).and_then(|c| c.as_dim()), Some(expected));
        }
        assert_eq!(k.layout(v), MemLayout::Tile { x: 32, y: 32, stride: 32 });
        assert_eq!(k.layout(n), MemLayout::Tile { x: 32, y: 32, stride: 32 });
    }
}
