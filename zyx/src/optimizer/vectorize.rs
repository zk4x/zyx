use crate::{dtype::Constant, graph::BOp, kernel::{Kernel, Op, OpId, Scope}};

/*impl Kernel {
    /// This function will unroll define[N] into N x define[1] ops,
    /// so that we can use scalars instead of arrays in registers.
    /// But it can also unrool define[16] into 4 x define[4] for float4 vectors, etc.
    /// May be better to put this in optimizer.
    pub fn unroll_defines(&mut self) {
        // TODO
    }
}*/

impl Kernel {
    pub fn vectorize(&mut self, vectorize_dim: usize) {
        let mut op_id = self.tail;
        let mut loop_stack = Vec::new();
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Loop { dim, scope } => {
                    let endloop_id = loop_stack.pop().unwrap();
                    if scope == Scope::Register && dim <= vectorize_dim {
                        self.vectorize_loop(op_id, endloop_id);
                    }
                }
                Op::EndLoop => {
                    loop_stack.push(op_id);
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    // Move ops that can be vectorized outside of the loop
    pub fn vectorize_loop(&mut self, loop_id: OpId, endloop_id: OpId) {
        println!("Vectorizing loop={loop_id}, endloop={endloop_id}");
        let mut op_id = loop_id;
        let Op::Loop { dim: loop_dim, scope } = self.ops[loop_id].op else { unreachable!() };
        debug_assert!(loop_dim < 258);
        let mut c_indices = Vec::new();
        for i in 0..loop_dim {
            let cid = self.insert_before(loop_id, Op::Const(Constant::idx(i as u64)));
            c_indices.push(cid);
        }
        self.ops[op_id].op = Op::Vectorize { ops: c_indices };
        // Put them into vectorize
        debug_assert_eq!(scope, Scope::Register);
        while op_id != endloop_id {
            //let op = &mut self.ops[op_id].op;
            match self.ops[op_id].op {
                Op::Load { index, .. } => {
                    if let Some(idx) = self.get_vectorized_index(index, loop_id) {
                        let Op::Load { index, vlen: len, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = idx;
                        *len *= loop_dim as u8;
                    }
                }
                Op::Store { index, .. } => {
                    if let Some(idx) = self.get_vectorized_index(index, loop_id) {
                        let Op::Store { index, vlen: len, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = idx;
                        *len *= loop_dim as u8;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        self.remove(endloop_id);
    }

    fn get_vectorized_index(&self, index: OpId, loop_id: OpId) -> Option<OpId> {
        //println!("Get vectorized index={index}, loop_id={loop_id}");
        if let Op::Binary { x, y, bop } = self.ops[index].op {
            if bop == BOp::Add {
                if y == loop_id {
                    return Some(x);
                } else if x == loop_id {
                    return Some(y);
                }
            }
        }
        None
    }
}
