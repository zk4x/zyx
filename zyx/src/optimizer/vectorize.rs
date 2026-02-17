use crate::{
    dtype::Constant, graph::BOp, kernel::{Kernel, Op, OpId}, shape::Dim
};

impl Kernel {
    pub fn vectorize_loads(&mut self) {

        // TODO for now this function ignores aliasing of stores and loads.
        // So later we need to make sure there are no aliasing issues

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
        let mut loads: Vec<Vec<(OpId, OpId, Dim, OpId)>> = Vec::new();
        loads.push(Vec::new());
        'a: while !op_id.is_null() {
            match self.ops[op_id].op {
                Loop { .. } => {
                    loads.push(Vec::new());
                }
                Load { src, index, .. } => {
                    println!("load index: {:?}", self.at(index));
                    match self.ops[index].op {
                        Mad { x, y, z } => {
                            if let Const(c) = self.ops[z].op {
                                loads.last_mut().unwrap().push((op_id, src, c.as_dim(), index));
                            }
                        }
                        Binary { x, y, bop } if bop == BOp::Add => {
                            if let Const(c) = self.ops[x].op {
                                loads.last_mut().unwrap().push((op_id, src, c.as_dim(), index));
                            }
                            if let Const(c) = self.ops[y].op {
                                loads.last_mut().unwrap().push((op_id, src, c.as_dim(), index));
                            }
                        }
                        Const(c) => {
                            loads.last_mut().unwrap().push((op_id, src, c.as_dim(), index));
                        }
                        _ => {}
                    }
                }
                EndLoop => {
                    if let Some(mut loads) = loads.pop() {
                        if !loads.is_empty() {
                            println!("{loads:?}");
                            // Check if constant offsets are continuous numbers
                            loads.sort_by_key(|(_, _, idx, _)| *idx);
                            let mut i = 0;
                            for load in &loads {
                                if load.2 != i {
                                    continue 'a;
                                }
                                i += 1;
                            }

                            // Check that all loads load from the same source
                            let mut source = loads[0].1;
                            for load in &loads {
                                if load.1 != source {
                                    continue 'a;
                                }
                            }

                            // Get the base index
                            let mut base_index;
                            match self.ops[loads[0].3].op {
                                Mad { x, y, z } => {
                                    if let Const(c) = self.ops[z].op {
                                        base_index = self.insert_before(loads[0].0, Binary { x, y, bop: BOp::Mul });
                                    } else {
                                        continue 'a;
                                    }
                                }
                                Binary { x, y, bop } if bop == BOp::Add => {
                                    if let Const(_) = self.ops[x].op {
                                        base_index = y;
                                    } else if let Const(_) = self.ops[y].op {
                                        base_index = x;
                                    } else {
                                        continue 'a;
                                    }
                                }
                                Const(c) => {
                                    base_index = OpId::NULL;
                                }
                                _ => {
                                    continue 'a;
                                }
                            }

                            // Now that we know offsets are continues, we can replace the loads with single vectorized load
                            if base_index == OpId::NULL {
                                base_index = self.insert_before(loads[0].0, Const(Constant::idx(0)));
                            }
                            let vload = self.insert_before(loads[0].0, Load { src: loads[0].1, index: base_index, vlen: loads.len() as u8 });
                            for (idx, load) in loads.iter().enumerate() {
                                self.ops[load.0].op = Devectorize { vec: vload, idx };
                            }
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
