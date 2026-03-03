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
    graph::BOp,
    kernel::{Kernel, MMADType, MMADims, MMALayout, Op, OpId, Scope},
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
        use Op::*;

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
                        let Loop { dim } = self.ops[k_loop_id].op else { unreachable!() };
                        if dim == 8 {
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
        use BOp::*;
        use Op::*;

        let &Store { dst: acc_id, x, index: store_idx, vlen: 1 } = self.at(store_id) else { return None };
        let (c_base_index, c_offset) = self.index_base_and_offset(store_idx, k_loop_id);
        if c_base_index.keys().any(|k| !k.is_null()) {
            return None;
        } // Accumulator does not have a base index

        let &Binary { x, mut y, bop: Add } = self.at(x) else { return None };
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
            let &Binary { x, y, bop: Mul } = self.at(x) else { return None };
            let &Load { src: a, index, vlen: 1 } = self.at(x) else { return None };
            let Define { dtype: a_dtype, .. } = self.ops[a].op else { unreachable!() };
            let (a_base_index, a_offset) = self.index_base_and_offset(index, k_loop_id);
            let &Load { src: b, index, vlen: 1 } = self.at(y) else { return None };
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
            // Version without cast
            todo!()
        }
    }

    fn index_base_and_offset(&self, index: OpId, k_loop_id: OpId) -> (Map<OpId, Dim>, usize) {
        //println!("Getting base index and offset of index={index}");
        //println!("{:?}", self.get_indices(index));

        let mut offset = 0;
        let indices = self.get_indices(index);
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
        self.debug();
        for store in stores {
            println!("{store:?}");
        }

        // A load
        let mut index = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        for (&i, &st) in &stores[0].a_index {
            let stride = self.insert_before(k_loop_id, Op::Const(Constant::idx(st as u64)));
            index = self.insert_before(k_loop_id, Op::Binary { x: stride, y: i, bop: BOp::Mul });
        }
        let a_load = self.insert_before(k_loop_id, Op::Load { src: stores[0].a, index, vlen: 2 });

        // B load
        let mut index = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        for (&i, &st) in &stores[0].b_index {
            let stride = self.insert_before(k_loop_id, Op::Const(Constant::idx(st as u64)));
            index = self.insert_before(k_loop_id, Op::Binary { x: stride, y: i, bop: BOp::Mul });
        }
        let b_load1 = self.insert_before(k_loop_id, Op::Load { src: stores[0].b, index, vlen: 1 });
        let b_load2 = self.insert_before(k_loop_id, Op::Load { src: stores[0].b, index, vlen: 1 });
        let b_load = self.insert_before(k_loop_id, Op::Vectorize { ops: vec![b_load1, b_load2] });

        // C load
        let index = self.insert_before(k_loop_id, Op::Const(Constant::idx(0)));
        let c_load = self.insert_before(k_loop_id, Op::Load { src: stores[0].c, index, vlen: 4 });

        self.insert_before(
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

        for store in stores {
            self.remove(store.store_id);
        }
    }

    /// Ensures threads have warp size compatible dimension
    fn warpize_threads(&mut self) -> bool {
        let mut local_dims = Vec::new();
        let mut local_loops = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { dim, scope } = self.ops[op_id].op {
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
            Op::Index { dim: local_dims[0] * n, scope: Scope::Local },
        );
        let y = self.insert_before(warp_loop, Op::Const(Constant::idx(n as u64)));
        self.ops[local_loops[0]].op = Op::Binary { x: warp_loop, y, bop: BOp::Div };
        let y = self.insert_before(warp_loop, Op::Const(Constant::idx(n as u64)));
        self.ops[local_loops[1]].op = Op::Binary { x: warp_loop, y, bop: BOp::Mod };

        true
    }
}
