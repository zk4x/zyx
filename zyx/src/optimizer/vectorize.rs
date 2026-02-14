use crate::{Map, Set, dtype::Constant, graph::BOp, kernel::{Kernel, Op, OpId, Scope}, shape::Dim};

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
    pub fn vectorize_loops(&mut self, vectorize_dim: usize) {
        let mut op_id = self.tail;
        let mut loop_stack = Vec::new();
        let mut has_inner_loops = false;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Loop { dim, scope } => {
                    let endloop_id = loop_stack.pop().unwrap();
                    if !has_inner_loops && scope == Scope::Register && dim <= vectorize_dim {
                        self.vectorize_loop(op_id, endloop_id);
                    } else {
                        has_inner_loops = true;
                    }
                }
                Op::EndLoop => {
                    loop_stack.push(op_id);
                    has_inner_loops = false;
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
        self.debug();
        println!("Vectorizing loop={loop_id}, endloop={endloop_id}");
        let mut op_id = loop_id;
        let Op::Loop { dim, scope } = self.ops[loop_id].op else { unreachable!() };
        debug_assert!(dim < 258);
        let loop_dim = dim as u8;
        let mut c_indices = Vec::new();
        for i in 0..loop_dim {
            let cid = self.insert_before(loop_id, Op::Const(Constant::idx(i as u64)));
            c_indices.push(cid);
        }
        self.ops[loop_id].op = Op::Vectorize { ops: c_indices };
        // Put them into vectorize
        debug_assert_eq!(scope, Scope::Register);

        let mut vectorized_ops = Set::default();
        vectorized_ops.insert(loop_id);
        while op_id != endloop_id {
            match &mut self.ops[op_id].op {
                Op::Load { vlen, .. } => {
                    *vlen = loop_dim;
                }
                Op::Store { vlen, .. } => {
                    *vlen = loop_dim;
                }
                ref op => {
                    let mut remapping = Map::default(); // becuase Rust is bad
                    for param in op.parameters().collect::<Vec<OpId>>() {
                        if !vectorized_ops.contains(&param) {
                            let op = Op::Vectorize { ops: vec![param; loop_dim as usize] };
                            let id = self.insert_before(loop_id, op);
                            remapping.insert(param, id);
                        }
                    }
                    self.ops[op_id].op.remap_params(&remapping);
                }
            }
            vectorized_ops.insert(op_id);
            op_id = self.next_op(op_id);
        }
        self.remove(endloop_id);
    }

    /// Searches the whole kernel. If it finds ops that can be groupped together, puts vectorize before and devectorize after them and groups (vectorizes) them.
    #[allow(unused)]
    pub fn vectorize_ops(&mut self) {
        let mut op_id = self.tail;
        let mut loop_stack = Vec::new();
        let mut no_inner_loops = true;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Loop { dim, scope } => {
                    let endloop_id = loop_stack.pop().unwrap();
                    if no_inner_loops {
                        self.vectorize_ops_in_loop(op_id, endloop_id);
                    } else {
                        no_inner_loops = false;
                    }
                }
                Op::EndLoop => {
                    loop_stack.push(op_id);
                    no_inner_loops = true;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn vectorize_ops_in_loop(&mut self, loop_id: OpId, endloop_id: OpId) {
        // A simple version is to use devectorize op and gradually keep looking for groups of ops to devectorize and last step would be to simply merge vectorize and devectorize ops together.
        // And merge vectorize ops with loads and devectorize ops with stores if the last stride is 1
        self.debug();
        println!("Vectorizing ops in loop={loop_id}, endloop_id={endloop_id}");

        let mut loads: Map<OpId, Vec<OpId>> = Map::default();
        let mut op_id = loop_id;
        while op_id != endloop_id {
            match self.ops[op_id].op {
                Op::Load { src, vlen, .. } => {
                    if vlen == 1 {
                        loads.entry(src).and_modify(|v| v.push(op_id)).or_insert_with(|| vec![op_id]);
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        for (src, loads) in loads {
            todo!()
        }

        self.debug();
        panic!();
    }

    pub fn vectorize_loads(&mut self) {
        use Op::*; use Scope::*;

        self.swap_commutative();
        self.reassociate_commutative();
        self.loop_invariant_code_motion();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.move_constants_to_beginning();

        self.debug();

        let mut op_id = self.head;
        let mut loads: Vec<Vec<(OpId, OpId, Dim)>> = Vec::new();
        loads.push(Vec::new());
        while !op_id.is_null() {

            match self.ops[op_id].op {
                Loop { .. } => {
                    loads.push(Vec::new());
                }
                Load { src, index, .. } => {
                    println!("load index: {:?}", self.at(index));
                    match self.ops[index].op {
                        Mad { x, y, z } => {
                            if let Const(c) = self.ops[z].op {
                                loads.last_mut().unwrap().push((op_id, src, c.as_dim()));
                            }
                        }
                        Const(c) => {
                            loads.last_mut().unwrap().push((op_id, src, c.as_dim()));
                        }
                        _ => {}
                    }
                }
                EndLoop => {
                    if let Some(loads) = loads.pop() {
                        if !loads.is_empty() {
                            println!("{loads:?}");
                            for load_id in loads {
                                //let Load { src, index, vlen } = self.ops[load_id].op else { unreachable!() };
                            }
                            todo!();
                        }
                    }
                }
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        todo!();
    }

    pub fn vectorize_stores(&mut self) {}
}
