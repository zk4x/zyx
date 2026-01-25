use crate::{
    Map, Set,
    backend::DeviceInfo,
    dtype::Constant,
    graph::BOp,
    kernel::{Kernel, Op, OpId, Scope},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopJamOpt {
    max_register_bytes: Dim,
}

impl LoopJamOpt {
    pub fn new(_kernel: &Kernel, dev_info: &DeviceInfo) -> (Self, u32, Vec<u32>) {
        (Self { max_register_bytes: dev_info.max_register_bytes }, 1, vec![0]) // 1, 64 loop jam
    }

    // It's complex :P
    #[must_use]
    pub fn apply_optimization(&self, _index: u32, kernel: &mut Kernel) -> bool {
        let jam_dim = 64;

        let mut jam_found;
        loop {
            jam_found = false;
            let mut active_defines: Vec<(OpId, Set<OpId>)> = Vec::new();
            let mut op_id = kernel.head;
            'a: while !op_id.is_null() {
                let next = kernel.next_op(op_id);
                match *kernel.at(op_id) {
                    Op::Loop { scope, .. } => {
                        if scope == Scope::Register {
                            active_defines.push((op_id, Set::default()));
                        }
                    }
                    Op::EndLoop => {
                        active_defines.pop();
                    }
                    Op::Define { scope, .. } => {
                        if scope == Scope::Register {
                            if let Some(define_level) = active_defines.last_mut() {
                                define_level.1.insert(op_id);
                            }
                        }
                    }
                    Op::Load { src, .. } => {
                        for (loop_id, define_ids) in &active_defines {
                            if define_ids.contains(&src) {
                                let Op::Loop { dim, scope } = kernel.ops[*loop_id].op else { unreachable!() };
                                debug_assert_eq!(scope, Scope::Register);
                                if dim <= jam_dim {
                                    let inner_loop_id = active_defines.last().unwrap().0;

                                    // Uncommenting this makes loop jam jam only past the first reduce loop,
                                    // otherwise it moves past the last reduce loop.
                                    /*for (id, _active_defines) in active_defines.iter().rev() {
                                        if id == loop_id {
                                            break;
                                        }
                                        inner_loop_id = *id;
                                    }*/

                                    if inner_loop_id == *loop_id {
                                        break;
                                    }
                                    if !kernel.loop_jam(*loop_id, inner_loop_id) {
                                        break 'a;
                                    };
                                    //println!("Active defines: {active_defines:?}");
                                    jam_found = true;
                                    kernel.constant_folding();
                                    kernel.dead_code_elimination();
                                    kernel.common_subexpression_elimination();
                                    kernel.delete_empty_loops();
                                    break 'a;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                op_id = next;
            }

            if !jam_found {
                break;
            }
        }

        true
    }
}

impl Kernel {
    /// Jam into loop. Yes, it's complex :P
    pub fn loop_jam(&mut self, jam_loop_id: OpId, inner_loop_id: OpId) -> bool {
        // If any pre loop op is load, we can't apply loop jam
        let mut op_id = jam_loop_id;
        while op_id != inner_loop_id {
            op_id = self.next_op(op_id);
            if self.at(op_id).is_load() {
                return false;
            }
        }

        let mut op_id = jam_loop_id;
        let mut loop_level = 0;
        let mut middle_loop_id = OpId::NULL;
        let mut end_middle_loop_id = OpId::NULL;
        let mut end_inner_loop_id = OpId::NULL;
        let mut inner_loop_level = None;
        let mut pre_loop_ops = Set::default();
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    if loop_level == 1 {
                        middle_loop_id = op_id;
                    }
                    if op_id == inner_loop_id {
                        inner_loop_level = Some(loop_level);
                    }
                    loop_level += 1;
                }
                Op::EndLoop => {
                    loop_level -= 1;
                    if let Some(inner_loop_level) = inner_loop_level {
                        if loop_level == inner_loop_level {
                            end_inner_loop_id = op_id;
                        }
                        if loop_level == 1 {
                            end_middle_loop_id = op_id;
                            break;
                        }
                    }
                }
                _ => {}
            }
            if loop_level == 1 {
                pre_loop_ops.insert(op_id);
            }
            op_id = self.next_op(op_id);
        }
        debug_assert_ne!(end_inner_loop_id, OpId::NULL);
        debug_assert_ne!(end_middle_loop_id, OpId::NULL);

        // TODO checks that between the middle and the inner loop there are no ops that depend on ops inside the pre loop
        let mut op_id = middle_loop_id;
        while op_id != inner_loop_id {
            if self.ops[op_id].op.parameters().any(|p| pre_loop_ops.contains(&p)) {
                return false;
            }
            op_id = self.next_op(op_id);
        }
        let mut op_id = end_inner_loop_id;
        while op_id != end_middle_loop_id {
            if self.ops[op_id].op.parameters().any(|p| pre_loop_ops.contains(&p)) {
                return false;
            }
            op_id = self.next_op(op_id);
        }

        //println!("Loop jam, jam_loop={jam_loop_id}, middle_loop={middle_loop_id}, inner_loop={inner_loop_id}");
        //println!("end_middle_loop={end_middle_loop_id}, pre_loop_ops={pre_loop_ops:?}");

        let Op::Loop { dim: jam_dim, scope } = self.ops[jam_loop_id].op else { unreachable!() };
        debug_assert_eq!(scope, Scope::Register);

        // Add constnat for dimension, will be used for indexing
        let const_jam_dim = self.insert_before(jam_loop_id, Op::Const(Constant::idx(jam_dim as u64)));

        // ***** Pre loop *****
        // Move all defines before the loop
        let mut defines = Set::default();
        let mut op_id = jam_loop_id;
        while op_id != middle_loop_id {
            op_id = self.next_op(op_id);
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                self.ops[op_id].op = Op::Define { dtype, scope, ro, len: len * jam_dim };
                defines.insert(op_id);
                self.move_op_before(op_id, jam_loop_id);
            }
        }

        // Reindex stores
        let mut op_id = jam_loop_id;
        while op_id != middle_loop_id {
            op_id = self.next_op(op_id);
            match *self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: jam_loop_id, bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = new_index;
                    }
                }
                _ => {}
            }
        }
        let end_pre_loop = self.insert_before(middle_loop_id, Op::EndLoop);

        // ***** Inner loop *****
        // Insert pre loop into inner loop and remap
        let mut remapping = Map::default();
        let mut op_id = jam_loop_id;
        let mut t_op_id = inner_loop_id;
        while op_id != end_pre_loop {
            let mut op = self.ops[op_id].op.clone();
            match self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { .. } => {}
                _ => {
                    op.remap_params(&remapping);
                    t_op_id = self.insert_after(t_op_id, op);
                    remapping.insert(op_id, t_op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        // Remap inner loop
        let mut op_id = t_op_id;
        //println!("Inner loop starting at {op_id}");
        let mut loop_level = 1;
        loop {
            op_id = self.next_op(op_id);
            self.ops[op_id].op.remap_params(&remapping);
            match self.ops[op_id].op {
                Op::Load { src, index, .. } => {
                    if defines.contains(&src) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index =
                            self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Load { index, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index =
                            self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Loop { .. } => loop_level += 1,
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        self.insert_before(op_id, Op::EndLoop);

        // ***** Post inner loop *****
        // Pre loop ops
        remapping.clear();
        let mut t_op_id = end_middle_loop_id;
        let mut op_id = jam_loop_id;
        while op_id != end_pre_loop {
            let mut op = self.ops[op_id].op.clone();
            match self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { .. } => {}
                _ => {
                    op.remap_params(&remapping);
                    t_op_id = self.insert_after(t_op_id, op);
                    remapping.insert(op_id, t_op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        // Remap post loop ops
        let mut op_id = t_op_id;
        //println!("Post loop starting at {op_id}");
        let mut loop_level = 1;
        loop {
            op_id = self.next_op(op_id);
            self.ops[op_id].op.remap_params(&remapping);
            match self.ops[op_id].op {
                Op::Load { src, index, .. } => {
                    if defines.contains(&src) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index =
                            self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Load { index, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index =
                            self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Loop { .. } => loop_level += 1,
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }

        #[cfg(debug_assertions)]
        self.verify();

        true
    }
}
