// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tiling pass: scalar row-major traffic becomes tile traffic.
//! (Under construction, one step at a time.)

use super::{Kernel, MemLayout, Op, OpId, RangeKind};
use crate::shape::Dim;

impl Kernel {
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let tilew: Dim = 32;
        let mut axes: [(OpId, Dim); 2] = [(OpId::NULL, 0); 2];
        for (op, x) in self.iter_unordered() {
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
        for (op, len) in axes.iter() {
            let short = self.insert_const_idx_before(*op, len / tilew);
            let Op::Range { axis, .. } = self.ops[*op].op else { return };
            self.ops[*op].op = Op::Range { axis, kind: RangeKind::Group(short) };
        }
        self.verify();
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::kernel::{Dev, MemLayout, Op, RangeKind};
    use super::Kernel;

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
