use crate::{
    kernel::{Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    pub fn vectorize_loads(&mut self) {
        use Op::*;

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
