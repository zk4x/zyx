// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

// For WMMA
// We need to have specific size of inner and local work sizes such
//
// 1. work_size.rs, the loads are permuted such that we can use tensor cores by putting
// the local and register loops after the inner reduce loop (in strides).
//
// 2. now we can replace the inner register loops with the tensor core instructions directly,
// nothing else to do, just verify the strides :D
//

use crate::{
    DType, Map,
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, Kernel, MMADType, MMADims, MMALayout, Op, OpId, Scope},
    shape::Dim,
};

#[derive(Debug)]
struct MMAStore {
    store_id: OpId,
    a: OpId,
    a_index: Map<OpId, Dim>,
    a_offset: Dim,
    a_dtype: DType,
    b: OpId,
    b_index: Map<OpId, Dim>,
    b_offset: Dim,
    b_dtype: DType,
    c: OpId,
    c_offset: Dim,
    c_dtype: DType,
}

impl Kernel {
    /// Finds loop trifectas and if possible, LICM moves these instructions before those loops
    /// and converts them into MMA instructions.
    pub fn fuse_mma(&mut self, dev_info: &DeviceInfo) {
        #[allow(clippy::enum_glob_use)]
        use Op::*;

        if !dev_info.tensor_cores {
            return;
        }

        self.unroll_loops(4);
        self.swap_commutative();
        //self.reassociate_commutative();
        self.loop_invariant_code_motion();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.move_constants_to_beginning();

        let mut stores = Vec::new();

        let mut op_id = self.head;
        let mut loop_ids = Vec::new();
        let mut mma_exists = false;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Loop { .. } => {
                    loop_ids.push(op_id);
                    stores.push(Vec::new());
                }
                EndLoop => {
                    if let Some(k_loop_id) = loop_ids.pop() {
                        if let Some(stores) = stores.pop() {
                            if !stores.is_empty() {
                                self.write_mma_op(&stores, k_loop_id);
                                mma_exists = true;
                            }
                        }
                    }
                }
                Store { .. } => {
                    if let Some(&k_loop_id) = loop_ids.last() {
                        let Loop { len, .. } = self.ops[k_loop_id].op else { unreachable!() };
                        if len == 8 {
                            if let Some(store_info) = self.mma_store_info(op_id, k_loop_id) {
                                stores.last_mut().unwrap().push(store_info);
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        if mma_exists {
            let can_warpize_threads = self.warpize_threads();
            assert!(can_warpize_threads);
        }
    }

    fn mma_store_info(&self, store_id: OpId, k_loop_id: OpId) -> Option<MMAStore> {
        #[allow(clippy::enum_glob_use)]
        use BOp::*;
        #[allow(clippy::enum_glob_use)]
        use Op::*;

        let &Store { dst: acc_id, x, index: store_idx, vlen: 1 } = self.at(store_id) else {
            return None;
        };
        let (c_base_index, c_offset) = self.index_base_and_offset(store_idx, k_loop_id);
        if c_base_index.keys().any(|k| !k.is_null()) {
            return None;
        } // Accumulator does not have a base index

        let &Binary { x, mut y, bop: Add } = self.at(x) else {
            return None;
        };
        let (src, index) = if let &Load { src, index, vlen: 1 } = self.at(x) {
            (src, index)
        } else if let &Load { src, index, vlen: 1 } = self.at(y) {
            y = x;
            (src, index)
        } else {
            return None;
        };

        if src != acc_id || index != store_idx {
            return None;
        }

        if let &Cast { x, dtype } = self.at(y) {
            let &Binary { x, y, bop: Mul } = self.at(x) else {
                return None;
            };
            let &Load { src: a, index, vlen: 1 } = self.at(x) else {
                return None;
            };
            let Define { dtype: a_dtype, .. } = self.ops[a].op else { unreachable!() };
            let (a_base_index, a_offset) = self.index_base_and_offset(index, k_loop_id);
            let &Load { src: b, index, vlen: 1 } = self.at(y) else {
                return None;
            };
            let Define { dtype: b_dtype, .. } = self.ops[b].op else { unreachable!() };
            let (b_base_index, b_offset) = self.index_base_and_offset(index, k_loop_id);

            Some(MMAStore {
                store_id,
                a,
                a_index: a_base_index,
                a_offset,
                a_dtype,
                b,
                b_index: b_base_index,
                b_offset,
                b_dtype,
                c: acc_id,
                c_offset,
                c_dtype: dtype,
            })
        } else {
            let x = y;
            // Version without cast
            let &Binary { x, y, bop: Mul } = self.at(x) else {
                return None;
            };
            let &Load { src: a, index, vlen: 1 } = self.at(x) else {
                return None;
            };
            let Define { dtype: a_dtype, .. } = self.ops[a].op else { unreachable!() };
            let (a_base_index, a_offset) = self.index_base_and_offset(index, k_loop_id);
            let &Load { src: b, index, vlen: 1 } = self.at(y) else {
                return None;
            };
            let Define { dtype: b_dtype, .. } = self.ops[b].op else { unreachable!() };
            let (b_base_index, b_offset) = self.index_base_and_offset(index, k_loop_id);

            Some(MMAStore {
                store_id,
                a,
                a_index: a_base_index,
                a_offset,
                a_dtype,
                b,
                b_index: b_base_index,
                b_offset,
                b_dtype,
                c: acc_id,
                c_offset,
                c_dtype: a_dtype,
            })
        }
    }

    fn index_base_and_offset(&self, index: OpId, k_loop_id: OpId) -> (Map<OpId, Dim>, Dim) {
        //println!("Getting base index and offset of index={index}");
        //println!("{:?}", self.get_indices(index));

        let mut offset: Dim = 0;
        let indices = self.get_strides(index);
        let mut new_indices = Map::default();
        for (loop_id, (_d, st)) in indices {
            if loop_id.is_null() {
                offset = st;
            } else if loop_id != k_loop_id {
                new_indices.insert(loop_id, st);
            }
        }

        (new_indices, offset)
    }

    fn write_mma_op(&mut self, stores: &[MMAStore], k_loop_id: OpId) {
        // A load
        // TODO we need to put the warp row with stride 1, while offset needs to be adjusted
        let mut idx = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        for (&i, &st) in &stores[0].a_index {
            let stride = self.insert_before(k_loop_id, Op::Const(Constant::idx(st as u64)));
            let y = self.insert_before(k_loop_id, Op::Binary { x: stride, y: i, bop: BOp::Mul });
            idx = self.insert_before(k_loop_id, Op::Binary { x: idx, y, bop: BOp::Add });
        }
        let a_load1 = self.insert_before(k_loop_id, Op::Load { src: stores[0].a, index: idx, vlen: 1 });
        let offset = self.insert_before(k_loop_id, Op::Const(Constant::idx(stores[1].a_offset as u64)));
        let index = self.insert_before(k_loop_id, Op::Binary { x: offset, y: idx, bop: BOp::Add });
        let a_load2 = self.insert_before(k_loop_id, Op::Load { src: stores[1].a, index, vlen: 1 });
        let offset = self.insert_before(k_loop_id, Op::Const(Constant::idx(stores[2].a_offset as u64)));
        let index = self.insert_before(k_loop_id, Op::Binary { x: offset, y: idx, bop: BOp::Add });
        let a_load3 = self.insert_before(k_loop_id, Op::Load { src: stores[2].a, index, vlen: 1 });
        let offset = self.insert_before(k_loop_id, Op::Const(Constant::idx(stores[3].a_offset as u64)));
        let index = self.insert_before(k_loop_id, Op::Binary { x: offset, y: idx, bop: BOp::Add });
        let a_load4 = self.insert_before(k_loop_id, Op::Load { src: stores[3].a, index, vlen: 1 });

        let a_load = self.insert_before(k_loop_id, Op::Vectorize { ops: vec![a_load1, a_load2, a_load3, a_load4] });

        // B load
        let mut idx = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        for (&i, &st) in &stores[0].b_index {
            let stride = self.insert_before(k_loop_id, Op::Const(Constant::idx(st as u64)));
            let y = self.insert_before(k_loop_id, Op::Binary { x: stride, y: i, bop: BOp::Mul });
            idx = self.insert_before(k_loop_id, Op::Binary { x: idx, y, bop: BOp::Add });
        }
        let b_load1 = self.insert_before(k_loop_id, Op::Load { src: stores[0].b, index: idx, vlen: 1 });
        let offset = self.insert_before(k_loop_id, Op::Const(Constant::idx(stores[1].b_offset as u64)));
        let index = self.insert_before(k_loop_id, Op::Binary { x: offset, y: idx, bop: BOp::Add });
        let b_load2 = self.insert_before(k_loop_id, Op::Load { src: stores[0].b, index, vlen: 1 });
        let b_load = self.insert_before(k_loop_id, Op::Vectorize { ops: vec![b_load1, b_load2] });

        // C load
        let index = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        let c_load = self.insert_before(k_loop_id, Op::Load { src: stores[0].c, index, vlen: 4 });

        let wmma_op = self.insert_before(
            k_loop_id,
            Op::WMMA {
                dims: MMADims::m16n8k8,
                layout: MMALayout::row_col,
                dtype: MMADType::f16_f16_f16_f32,
                c: c_load,
                a: a_load,
                b: b_load,
            },
        );
        self.insert_after(wmma_op, Op::Store { dst: stores[0].c, x: wmma_op, index, vlen: 4 });

        for store in stores {
            self.remove_op(store.store_id);
        }

        //self.debug();
        //todo!();
    }

    /// Ensures threads have warp size compatible dimension
    fn warpize_threads(&mut self) -> bool {
        let mut local_dims = Vec::new();
        let mut local_loops = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len: dim, scope, axis } = self.ops[op_id].op {
                if scope == Scope::Local {
                    local_dims.push(dim);
                    local_loops.push(op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        if local_loops.len() < 2 {
            return false;
        }

        // For now
        let n = 4;

        if local_dims[1] != n {
            return false;
        }

        let warp_loop = self.insert_before(
            local_loops[0],
            Op::Index { len: local_dims[0] * n, scope: Scope::Local, axis: 0 },
        );
        let y = self.insert_before(warp_loop, Op::Const(Constant::idx(n as u64)));
        self.ops[local_loops[0]].op = Op::Binary { x: warp_loop, y, bop: BOp::Div };
        let y = self.insert_before(warp_loop, Op::Const(Constant::idx(n as u64)));
        self.ops[local_loops[1]].op = Op::Binary { x: warp_loop, y, bop: BOp::Mod };

        true
    }
}
