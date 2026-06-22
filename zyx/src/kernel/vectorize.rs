// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map, Set,
    backend::DeviceInfo,
    kernel::{Kernel, MemLayout, Op, OpId},
    shape::Dim,
};

#[derive(Debug)]
struct LoadInfo {
    id: OpId,
    index: OpId,
}

#[derive(Debug)]
struct StoreInfo {
    id: OpId,
    index: OpId,
    x: OpId,
}

impl Kernel {
    pub(crate) fn opt_vectorize(&self, dev_info: &DeviceInfo) -> (Optimization, usize) {
        let supported_lens = if dev_info.has_vector_ops { vec![2, 4] } else { vec![] };
        (Optimization::Vectorize { supported_lens, vectorize_ops: dev_info.has_vector_ops }, 1)
    }

    /// Vectorize loads.
    ///
    /// Combines multiple loads into vectorized operations for better performance.
    /// `supported_lens` is the list of vector element counts the target device supports.
    /// TODO for now this function ignores aliasing of stores and loads.
    pub fn vectorize_loads(&mut self, supported_lens: &[u8]) {
        let mut op_id = self.head;
        // Map: src id -> LoadInfo
        let mut loads: Vec<Map<OpId, Vec<LoadInfo>>> = Vec::new();
        loads.push(Map::default());
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loads.push(Map::default());
                }
                Op::Load { src, index, layout } => {
                    if layout == MemLayout::Scalar {
                        loads
                            .last_mut()
                            .unwrap()
                            .entry(src)
                            .and_modify(|e| e.push(LoadInfo { id: op_id, index }))
                            .or_insert_with(|| vec![LoadInfo { id: op_id, index }]);
                    }
                }
                Op::EndLoop => self.verify_and_apply_vectorization(&mut loads, supported_lens),
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        self.verify_and_apply_vectorization(&mut loads, supported_lens);
    }

    fn verify_and_apply_vectorization(&mut self, loads: &mut Vec<Map<OpId, Vec<LoadInfo>>>, supported_lens: &[u8]) {
        if let Some(loads) = loads.pop() {
            for (src, mut loads) in loads {
                if !supported_lens.contains(&(loads.len() as u8)) {
                    continue;
                }

                loads.sort_unstable_by_key(|x| self.get_strides(x.index).len());

                let mut base_index = None;
                let mut offset_order: Vec<Dim> = Vec::new();
                let vec_len = loads.len() as Dim;
                for (base_idx, (_, vl)) in self.get_strides(loads[0].index) {
                    if vl != vec_len {
                        continue;
                    }
                    let mut offsets: Set<Dim> = (0..vec_len).collect();
                    offset_order.clear();

                    if loads[1..].iter().all(|x| {
                        let strides = self.get_strides(x.index);
                        strides.iter().any(|(&idx, (_, st))| idx == base_idx && *st == vec_len)
                            && strides.iter().any(|(&idx, (_, st))| {
                                let found = idx.is_null() && offsets.remove(st);
                                if found {
                                    offset_order.push(*st);
                                }
                                found
                            })
                    }) {
                        if offsets.remove(&0) {
                            base_index = Some(base_idx);
                            break;
                        }
                    }
                }

                // Now that we know offsets are continues, we can replace the loads with single vectorized load
                if base_index.is_some() {
                    let vload = self.insert_before(
                        loads[0].id,
                        Op::Load { src, index: loads[0].index, layout: MemLayout::Vector(vec_len as u16) },
                    );
                    self.ops[loads[0].id].op = Op::Devectorize { vec: vload, idx: 0 };
                    for (load, &off) in loads[1..].iter().zip(&offset_order) {
                        self.ops[load.id].op = Op::Devectorize { vec: vload, idx: off as usize };
                    }
                }
            }
        }
    }

    /// Vectorize stores.
    ///
    /// Combines multiple scalar stores into a single vectorized store for better performance.
    /// `supported_lens` is the list of vector element counts the target device supports.
    pub fn vectorize_stores(&mut self, supported_lens: &[u8]) {
        let mut op_id = self.head;
        let mut stores: Vec<Map<OpId, Vec<StoreInfo>>> = Vec::new();
        stores.push(Map::default());
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    stores.push(Map::default());
                }
                Op::Store { dst, x, index, layout } => {
                    if layout == MemLayout::Scalar {
                        stores
                            .last_mut()
                            .unwrap()
                            .entry(dst)
                            .and_modify(|e| e.push(StoreInfo { id: op_id, index, x }))
                            .or_insert_with(|| vec![StoreInfo { id: op_id, index, x }]);
                    }
                }
                Op::EndLoop => self.verify_and_apply_store_vectorization(&mut stores, supported_lens),
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        self.verify_and_apply_store_vectorization(&mut stores, supported_lens);
    }

    fn verify_and_apply_store_vectorization(&mut self, stores: &mut Vec<Map<OpId, Vec<StoreInfo>>>, supported_lens: &[u8]) {
        if let Some(stores) = stores.pop() {
            for (dst, mut stores) in stores {
                if !supported_lens.contains(&(stores.len() as u8)) {
                    continue;
                }

                stores.sort_unstable_by_key(|x| self.get_strides(x.index).len());

                let mut base_index = None;
                let mut offset_order: Vec<Dim> = Vec::new();
                let vec_len = stores.len() as Dim;
                for (base_idx, (_, vl)) in self.get_strides(stores[0].index) {
                    if vl != vec_len {
                        continue;
                    }
                    let mut offsets: Set<Dim> = (0..vec_len).collect();
                    offset_order.clear();

                    if stores[1..].iter().all(|x| {
                        let strides = self.get_strides(x.index);
                        strides.iter().any(|(&idx, (_, st))| idx == base_idx && *st == vec_len)
                            && strides.iter().any(|(&idx, (_, st))| {
                                let found = idx.is_null() && offsets.remove(st);
                                if found {
                                    offset_order.push(*st);
                                }
                                found
                            })
                    }) {
                        if offsets.remove(&0) {
                            base_index = Some(base_idx);
                            break;
                        }
                    }
                }

                // Now that we know offsets are contiguous, replace scalar stores with a single vectorized store
                if base_index.is_some() {
                    // Build vector values at correct offset positions
                    let mut vec_values = Vec::with_capacity(vec_len as usize);
                    vec_values.resize(vec_len as usize, OpId::NULL);
                    vec_values[0] = stores[0].x;
                    for (store, &off) in stores[1..].iter().zip(&offset_order) {
                        vec_values[off as usize] = store.x;
                    }

                    let vstore = self.insert_before(stores[0].id, Op::Vectorize { ops: vec_values });
                    self.ops[stores[0].id].op =
                        Op::Store { dst, x: vstore, index: stores[0].index, layout: MemLayout::Vector(vec_len as u16) };
                    for store in &stores[1..] {
                        self.remove_op(store.id);
                    }
                }
            }
        }
    }

    /// Walk backward, find Vectorize[X0..Xn] where all Xi are the same compute op.
    /// Replace Vectorize with compute_op(vectorize[inputs_of_Xi]),
    /// replace the first Xi with that vectorize of inputs, and keep walking.
    pub fn vectorize_ops_backward(&mut self, supported_lens: &[u8]) {
        let mut op_id = self.tail;
        while !op_id.is_null() {
            let ops = match &self.ops[op_id].op {
                Op::Vectorize { ops } => ops.clone(),
                _ => {
                    op_id = self.prev_op(op_id);
                    continue;
                }
            };

            let n = ops.len();
            if n < 2 || !supported_lens.contains(&(n as u8)) {
                op_id = self.prev_op(op_id);
                continue;
            }

            match &self.ops[ops[0]].op {
                Op::Unary { uop, .. } => {
                    let uop = *uop;
                    let mut sources = Vec::with_capacity(n);
                    for &sub in &ops {
                        match &self.ops[sub].op {
                            Op::Unary { x, uop: u } if *u == uop => sources.push(*x),
                            _ => {
                                sources.clear();
                                break;
                            }
                        }
                    }
                    if !sources.is_empty() && !sources.iter().any(|s| ops.contains(s)) {
                        self.ops[ops[0]].op = Op::Vectorize { ops: sources };
                        self.ops[op_id].op = Op::Unary { x: ops[0], uop };
                    }
                }
                Op::Cast { dtype, .. } => {
                    let dtype = *dtype;
                    let mut sources = Vec::with_capacity(n);
                    for &sub in &ops {
                        match &self.ops[sub].op {
                            Op::Cast { x, dtype: d } if *d == dtype => sources.push(*x),
                            _ => {
                                sources.clear();
                                break;
                            }
                        }
                    }
                    if !sources.is_empty() && !sources.iter().any(|s| ops.contains(s)) {
                        self.ops[ops[0]].op = Op::Vectorize { ops: sources };
                        self.ops[op_id].op = Op::Cast { x: ops[0], dtype };
                    }
                }
                Op::Binary { bop, .. } => {
                    let bop = *bop;
                    let mut xs = Vec::with_capacity(n);
                    let mut ys = Vec::with_capacity(n);
                    for &sub in &ops {
                        match &self.ops[sub].op {
                            Op::Binary { x, y, bop: b } if *b == bop => {
                                xs.push(*x);
                                ys.push(*y);
                            }
                            _ => {
                                xs.clear();
                                break;
                            }
                        }
                    }
                    if !xs.is_empty() && !xs.iter().any(|x| ops.contains(x)) && !ys.iter().any(|y| ops.contains(y)) {
                        let vy = self.insert_before(ops[0], Op::Vectorize { ops: ys });
                        self.ops[ops[0]].op = Op::Vectorize { ops: xs };
                        self.ops[op_id].op = Op::Binary { x: ops[0], y: vy, bop };
                    }
                }
                _ => {}
            }

            op_id = self.prev_op(op_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DType,
        kernel::{BOp, DeviceId, Kernel, MemLayout, Op, Scope, UOp},
    };

    #[test]
    fn vectorize_ops_and_constfold_clears_vectorize_devectorize() {
        let mut k = Kernel::new(DeviceId::AUTO);

        let src = k.define(DType::F32, Scope::Global, true, 16);
        let dst = k.define(DType::F32, Scope::Global, false, 16);
        let g0 = k.gidx(0, 4);
        let two = k.const_idx(2u32);
        let offset = k.binary(g0, two, BOp::BitShiftLeft);
        let vec_load = k.load(src, offset, MemLayout::Vector(4));
        let [s0, s1, s2, s3] = k.devectorize::<4>(vec_load);
        let c0 = k.unary(s0, UOp::Cos);
        let c1 = k.unary(s1, UOp::Cos);
        let c2 = k.unary(s2, UOp::Cos);
        let c3 = k.unary(s3, UOp::Cos);
        let vec = k.vectorize(vec![c0, c1, c2, c3]);
        k.store(dst, vec, offset, MemLayout::Vector(4));

        k.vectorize_ops_backward(&[4]);
        k.constant_folding();
        k.dead_code_elimination();

        let mut op_id = k.head;
        while !op_id.is_null() {
            match k.ops[op_id].op {
                Op::Vectorize { .. } | Op::Devectorize { .. } => {
                    panic!("Found Vectorize/Devectorize op at {op_id} after passes");
                }
                _ => {}
            }
            op_id = k.next_op(op_id);
        }
    }
}
