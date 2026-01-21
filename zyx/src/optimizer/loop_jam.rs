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
        // TODO make it work per loop?
        (Self { max_register_bytes: dev_info.max_register_bytes }, 1, vec![0]) // 1, 64 loop jam
    }

    // It's complex :P
    #[must_use]
    pub fn apply_optimization(&self, _index: u32, kernel: &mut Kernel) -> bool {
        let jam_dim = 32; //[1, 64][index as usize]; // TODO just uncomment this after other things are done

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
                                    let mut inner_loop_id = active_defines.last().unwrap().0;
                                    for (id, _active_defines) in active_defines.iter().rev() {
                                        /*for &def_op in active_defines {
                                            let Op::Define { len, dtype, .. } = kernel.ops[def_op] else {
                                                unreachable!()
                                            };
                                            if (len * dim) * dtype.byte_size() as Dim > self.max_register_bytes as Dim {
                                                // wold overflow the register space
                                                break 'a;
                                            }
                                        }*/
                                        if id == loop_id {
                                            break;
                                        }
                                        inner_loop_id = *id;
                                    }
                                    if inner_loop_id == *loop_id {
                                        break;
                                    }
                                    if !kernel.loop_jam(*loop_id, inner_loop_id) {
                                        break 'a;
                                    };
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
        //println!("Loop jam, jam_loop={jam_loop_id}, inner_loop={inner_loop_id}");

        let mut op_id = jam_loop_id;
        while op_id != inner_loop_id {
            op_id = self.next_op(op_id);
            if self.at(op_id).is_load() {
                return false;
            }
        }

        let Op::Loop { dim: jam_dim, scope } = self.ops[jam_loop_id].op else { unreachable!() };
        debug_assert_eq!(scope, Scope::Register);

        // Add constnat for dimension, will be used for indexing
        let const_jam_dim = self.insert_before(jam_loop_id, Op::Const(Constant::idx(jam_dim as u64)));

        // ***** Pre loop *****
        // Move all defines before the loop
        let mut defines = Set::default();
        let mut op_id = jam_loop_id;
        while op_id != inner_loop_id {
            op_id = self.next_op(op_id);
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                self.ops[op_id].op = Op::Define { dtype, scope, ro, len: len * jam_dim };
                defines.insert(op_id);
                self.move_op_before(op_id, jam_loop_id);
            }
        }

        // Handle stores
        let mut op_id = jam_loop_id;
        while op_id != inner_loop_id {
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
        let end_pre_loop = self.insert_before(inner_loop_id, Op::EndLoop);

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
        let mut t_op_id = op_id;
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
