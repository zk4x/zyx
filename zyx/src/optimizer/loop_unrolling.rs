use crate::{
    Map,
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope},
};
use nanoserde::{DeBin, SerBin};

/// loop unrolling
#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopUnrollingOpt {}

impl LoopUnrollingOpt {
    pub fn new(_kernel: &Kernel) -> (Self, u32) {
        (Self {}, 3)
    }

    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = [1, 4, 32][index as usize]; // TODO just uncomment this after other things are done
        let mut endloop_ids = Vec::new();
        let mut i = kernel.order.len();
        while i > 0 {
            i -= 1;
            let loop_id = kernel.order[i];
            if kernel.ops[loop_id] == Op::EndLoop {
                endloop_ids.push(loop_id);
            }
            if let Op::Loop { dim, scope } = kernel.ops[loop_id] {
                let endloop_id = endloop_ids.pop().unwrap();
                if scope == Scope::Register && dim <= unroll_dim && kernel.order.len() * dim < 10000 {
                    kernel.ops[loop_id] = Op::Const(Constant::idx(0));
                    let endloop_i = kernel.order.iter().rposition(|op_id| *op_id == endloop_id).unwrap();
                    let loop_order: &[OpId] = &kernel.order[i + 1..endloop_i];
                    let mut order = Vec::with_capacity(loop_order.len() * (dim - 1));
                    for idx in 1..dim {
                        let mut new_ops_map = Map::default();
                        let new_op_id = kernel.ops.push(Op::Const(Constant::idx(idx as u64)));
                        new_ops_map.insert(loop_id, new_op_id);
                        order.push(new_op_id);
                        for &op_id in loop_order {
                            let mut op = kernel.ops[op_id].clone();
                            for param in op.parameters_mut() {
                                if let Some(&new_param) = new_ops_map.get(param) {
                                    *param = new_param;
                                }
                            }
                            let new_op_id = kernel.ops.push(op);
                            new_ops_map.insert(op_id, new_op_id);
                            order.push(new_op_id);
                        }
                    }
                    kernel.ops.remove(endloop_id);
                    kernel.order.splice(endloop_i..=endloop_i, order);
                }
            }
        }
        #[cfg(debug_assertions)]
        kernel.verify();
        true
    }
}
