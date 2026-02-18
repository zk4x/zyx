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
    DType, Map, graph::BOp, kernel::{Kernel, Op, OpId, Scope}, shape::Dim
};

#[derive(Debug)]
struct MMAStore {
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
    pub fn fuse_mma(&mut self) {
        use Op::*;
        use Scope::*;

        self.unroll_loops(4);
        self.swap_commutative();
        //self.reassociate_commutative();
        self.loop_invariant_code_motion();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.move_constants_to_beginning();

        self.debug();

        let mut stores = Vec::new();

        let mut op_id = self.head;
        let mut loop_ids = Vec::new();
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Loop { .. } => {
                    loop_ids.push(op_id);
                    stores.push(Vec::new());
                }
                EndLoop => {
                    if let Some(k_loop_id) = loop_ids.pop() {
                        if let Some(stores) = stores.pop() {
                            self.write_mma_op(&stores, k_loop_id);
                        }
                    }
                }
                Store { .. } => {
                    if let Some(&k_loop_id) = loop_ids.last() {
                        let Loop { dim, scope } = self.ops[k_loop_id].op else { unreachable!() };
                        if scope == Register && dim == 16 {
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
    }

    fn mma_store_info(&self, store_id: OpId, k_loop_id: OpId) -> Option<MMAStore> {
        use BOp::*;
        use Op::*;

        let &Store { dst: acc_id, x, index: store_idx, vlen: 1 } = self.at(store_id) else { return None };
        let (c_base_index, c_offset) = self.index_base_and_offset(store_idx, k_loop_id);
        if c_base_index.keys().any(|k| !k.is_null()) { return None } // Accumulator does not have a base index

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
        for store in stores {
            println!("{store:?}");
        }
    }
}
