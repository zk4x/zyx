// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map, Set,
    kernel::{Kernel, MemLayout, Op, OpId},
    shape::Dim,
};

#[derive(Debug)]
struct LoadInfo {
    id: OpId,
    index: OpId,
}

impl Kernel {
    pub(crate) fn opt_vectorize_loads(&self) -> (Optimization, usize) {
        (Optimization::VectorizeLoads, 1)
    }

    /// Vectorize loads.
    ///
    /// Combines multiple loads into vectorized operations for better performance.
    /// TODO for now this function ignores aliasing of stores and loads.
    pub fn vectorize_loads(&mut self) {
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
                Op::EndLoop => self.fun_name(&mut loads),
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        self.fun_name(&mut loads);
    }

    fn fun_name(&mut self, loads: &mut Vec<Map<OpId, Vec<LoadInfo>>>) {
        if let Some(loads) = loads.pop() {
            for (src, mut loads) in loads {
                if ![2, 4, 8, 16, 32].contains(&loads.len()) {
                    continue;
                }

                loads.sort_unstable_by_key(|x| self.get_strides(x.index).len());

                let mut base_index = None;
                let vec_len = loads.len() as Dim;
                for (base_idx, (_, vl)) in self.get_strides(loads[0].index) {
                    if vl != vec_len {
                        continue;
                    }
                    let mut offsets: Set<Dim> = (0..vec_len).collect();

                    if loads[1..].iter().all(|x| {
                        let strides = self.get_strides(x.index);
                        strides.iter().any(|(&idx, (_, st))| idx == base_idx && *st == vec_len)
                            && strides.iter().any(|(&idx, (_, st))| idx.is_null() && offsets.remove(st))
                    }) {
                        if offsets.remove(&0) {
                            base_index = Some(base_idx);
                            break;
                        }
                    }
                }

                // Now that we know offsets are continues, we can replace the loads with single vectorized load
                if let Some(base_index) = base_index {
                    let vload = self.insert_before(
                        loads[0].id,
                        Op::Load { src, index: base_index, layout: MemLayout::Vector(vec_len as u8) },
                    );
                    for (idx, load) in loads.iter().enumerate() {
                        self.ops[load.id].op = Op::Devectorize { vec: vload, idx };
                    }
                }
            }
        }
    }

    #[allow(unused)]
    pub(crate) fn vectorize_ops(&mut self) {
        todo!()
    }

    #[allow(unused)]
    pub(crate) fn vectorize_stores(&mut self) {
        todo!()
    }
}
