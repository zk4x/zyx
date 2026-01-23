use crate::{
    Map,
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
};
use nanoserde::{DeBin, SerBin};

/// loop unrolling
#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopUnrollOpt {}

impl LoopUnrollOpt {
    pub fn new(_kernel: &Kernel) -> (Self, u32, Vec<u32>) {
        (Self {}, 1, vec![0])
    }

    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = [32, 8][index as usize];
        kernel.unroll_loops(unroll_dim);
        true
    }
}

impl Kernel {
    pub fn unroll_loops(&mut self, unroll_dim: usize) {
        let mut endloop_ids = Vec::new();
        let mut op_id = self.tail;
        while !op_id.is_null() {
            if self.ops[op_id].op == Op::EndLoop {
                endloop_ids.push(op_id);
            }
            if let Op::Loop { dim, scope } = self.ops[op_id].op {
                let endloop_id = endloop_ids.pop().unwrap();
                if scope == Scope::Register && dim <= unroll_dim && self.ops.len().0 < 5_000 {
                    self.unroll_loop(op_id, endloop_id, dim);
                }
            }
            op_id = self.prev_op(op_id);
        }
    }

    pub fn unroll_constant_loops(&mut self) {
        let mut endloop_ids = Vec::new();
        let mut op_id = self.tail;
        let mut constant_loops = vec![true];
        while !op_id.is_null() {
            let prev = self.prev_op(op_id);
            match self.ops[op_id].op {
                Op::EndLoop => {
                    endloop_ids.push(op_id);
                    constant_loops.push(true);
                }
                Op::Loop { dim, scope } => {
                    let endloop_id = endloop_ids.pop().unwrap();
                    //println!("Loop {op_id} constant={constant_loops:?}");
                    let is_const = constant_loops.pop().unwrap();
                    if !is_const {
                        if let Some(inner_loop) = constant_loops.last_mut() {
                            *inner_loop = false;
                        }
                    }
                    if scope == Scope::Register {
                        if dim == 1 || (is_const && dim < 500) {
                            self.unroll_loop(op_id, endloop_id, dim);
                        }
                    }
                }
                Op::Store { dst, .. } => {
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                Op::Load { src, .. } => {
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    if scope != Scope::Register {
                        *constant_loops.last_mut().unwrap() = false;
                    }
                }
                _ => {}
            }
            op_id = prev;
        }
    }

    pub fn unroll_loop(&mut self, loop_id: OpId, endloop_id: OpId, dim: usize) {
        //println!("Unrolling loop={loop_id}, end={endloop_id}, dim={dim}");
        self.ops[loop_id].op = Op::Const(Constant::idx(0));
        let last_loop_op = self.prev_op(endloop_id);

        for idx in 1..dim {
            let mut new_ops_map = Map::default();
            let idx_op = self.insert_before(endloop_id, Op::Const(Constant::idx(idx as u64)));
            new_ops_map.insert(loop_id, idx_op);

            let mut op_id = self.next_op(loop_id);
            loop {
                let mut op = self.ops[op_id].op.clone();
                for param in op.parameters_mut() {
                    if let Some(&new_param) = new_ops_map.get(param) {
                        *param = new_param;
                    }
                }
                let new_op_id = self.insert_before(endloop_id, op);
                new_ops_map.insert(op_id, new_op_id);

                if op_id == last_loop_op {
                    break;
                }
                op_id = self.next_op(op_id);
            }
        }
        self.remove(endloop_id);

        #[cfg(debug_assertions)]
        self.verify();
    }
}
