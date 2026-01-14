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
        (Self {}, 2, vec![0, 1])
    }

    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = [16, 16][index as usize]; //[1, 4, 32][index as usize]; // TODO just uncomment this after other things are done
        kernel.unroll_loops(unroll_dim);
        true
    }
}

impl Kernel {
    pub fn unroll_loops(&mut self, unroll_dim: usize) {
        /*let mut endloop_ids = Vec::new();
        let mut i = self.order.len();
        while i > 0 {
            i -= 1;
            let loop_id = self.order[i];
            if self.ops[loop_id] == Op::EndLoop {
                endloop_ids.push(loop_id);
            }
            if let Op::Loop { dim, scope } = self.ops[loop_id] {
                let endloop_id = endloop_ids.pop().unwrap();
                if scope == Scope::Register && dim <= unroll_dim && self.order.len() * dim < 50000 {
                    self.unroll_loop(i, loop_id, endloop_id, dim);
                }
            }
        }*/
        #[cfg(debug_assertions)]
        self.verify();
    }

    pub fn unroll_loop(&mut self, i: usize, loop_id: OpId, endloop_id: OpId, dim: usize) {
        /*self.ops[loop_id] = Op::Const(Constant::idx(0));
        let endloop_i = self.order.iter().rposition(|op_id| *op_id == endloop_id).unwrap();
        let loop_order: &[OpId] = &self.order[i + 1..endloop_i];
        let mut order = Vec::with_capacity(loop_order.len() * (dim - 1));
        for idx in 1..dim {
            let mut new_ops_map = Map::default();
            let new_op_id = self.ops.push(Op::Const(Constant::idx(idx as u64)));
            new_ops_map.insert(loop_id, new_op_id);
            order.push(new_op_id);
            for &op_id in loop_order {
                let mut op = self.ops[op_id].clone();
                for param in op.parameters_mut() {
                    if let Some(&new_param) = new_ops_map.get(param) {
                        *param = new_param;
                    }
                }
                let new_op_id = self.ops.push(op);
                new_ops_map.insert(op_id, new_op_id);
                order.push(new_op_id);
            }
        }
        self.ops.remove(endloop_id);
        self.order.splice(endloop_i..=endloop_i, order);*/
    }
}
