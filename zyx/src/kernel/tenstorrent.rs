// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{Kernel, Op, OpId},
    shape::Dim,
};

fn round_up(len: Dim, multiple: Dim) -> Dim {
    let rem = len % multiple;
    if rem == 0 { 0 } else { multiple - rem }
}

impl Kernel {
    pub(crate) fn opt_tenstorrent_pad(&mut self) {
        let mut gidxs: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::GroupIndex { len, axis } = self.at(op_id) {
                gidxs.push((op_id, axis, len));
            }
            // Can't run this optimization on kernel that already has local indices
            if let Op::LocalIndex { .. } = self.at(op_id) {
                return;
            }
            op_id = self.next_op(op_id);
        }
        gidxs.sort_by_key(|&(_, axis, _)| axis);
        gidxs.dedup_by_key(|&mut (_, axis, _)| axis);

        match gidxs.len() {
            0 | 2 => {
                for &(id, _axis, len) in &gidxs {
                    let pad = round_up(len, 32);
                    if pad > 0 {
                        self.pad_index(id, pad);
                    }
                }
            }
            1 => {
                let (id, _axis, len) = gidxs[0];
                let pad = round_up(len, 1024);
                if pad > 0 {
                    self.pad_index(id, pad);
                }
                let new_len = if let Op::GroupIndex { len, .. } = self.at(id) {
                    *len
                } else {
                    unreachable!()
                };
                let f1 = (new_len as f64).sqrt() as Dim;
                let f1 = (2..=f1).rev().find(|&f| new_len % f == 0).unwrap_or(1);
                if f1 <= 1 || f1 == new_len {
                    return;
                }
                let f2 = new_len / f1;
                self.split_dim(id, vec![Op::GroupIndex { len: f1, axis: 0 }, Op::GroupIndex { len: f2, axis: 1 }]);
            }
            3 => {
                let (last_id, _last_axis, last_len) = gidxs[2];
                for &(id, _axis, len) in &gidxs[..2] {
                    let pad = round_up(len, 32);
                    if pad > 0 {
                        self.pad_index(id, pad);
                    }
                }
                let len_const = self.insert_before(last_id, Op::Const(Constant::idx(last_len)));
                self.ops[last_id].op = Op::Loop { len: len_const };
                self.push_back(Op::EndLoop);
            }
            _ => {}
        }
    }
}
