use nanoserde::{DeBin, SerBin};
use crate::{Map, Set, backend::DeviceInfo, dtype::Constant, graph::BOp, kernel::{Kernel, Op, OpId, Scope}, shape::Dim};

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopJamOpt {
    max_register_bytes: Dim,
}

impl LoopJamOpt {
    pub fn new(_kernel: &Kernel, dev_info: &DeviceInfo) -> (Self, u32) {
        (Self { max_register_bytes: dev_info.max_register_bytes }, 2) // 1, 64 loop jam
    }

    // It's complex :P
    #[must_use]
    pub fn apply_optimization(&self, _index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = 0; //[1, 64][index as usize]; // TODO just uncomment this after other things are done

        let mut jam_found;
        loop {
            jam_found = false;
            let mut active_defines: Vec<(OpId, Set<OpId>)> = Vec::new();
            'a: for &op_id in &kernel.order {
                match kernel.ops[op_id] {
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
                                let Op::Loop { dim, scope } = kernel.ops[*loop_id] else { unreachable!() };
                                debug_assert_eq!(scope, Scope::Register);
                                if dim <= unroll_dim {
                                    let mut inner_loop_id = active_defines.last().unwrap().0;
                                    for (id, active_defines) in active_defines.iter().rev() {
                                        for &def_op in active_defines {
                                            let Op::Define { len, dtype, .. } = kernel.ops[def_op] else {
                                                unreachable!()
                                            };
                                            if (len * dim) * dtype.byte_size() as Dim > self.max_register_bytes as Dim {
                                                // wold overflow the register space
                                                break 'a;
                                            }
                                        }
                                        if id == loop_id {
                                            break;
                                        }
                                        inner_loop_id = *id;
                                    }
                                    if inner_loop_id == *loop_id {
                                        break;
                                    }
                                    kernel.loop_jam(*loop_id, inner_loop_id);
                                    jam_found = true;
                                    break 'a;
                                }
                            }
                        }
                    }
                    _ => {}
                }
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
    pub fn loop_jam(&mut self, jam_loop_id: OpId, inner_loop_id: OpId) {
        self.debug();
        println!("Loop jam, jam_loop={jam_loop_id}, inner_loop={inner_loop_id}");

        let mut pre_loop_ops = Vec::new();
        let mut inner_loop_ops = Vec::new();
        let mut post_loop_ops: Vec<OpId> = Vec::new();

        let mut stage = 0;
        let mut inner_loop_level = 0;
        let mut splice_ops_len = 0;

        let mut jam_loop_i = 0;
        for (i, &op_id) in self.order.iter().enumerate() {
            if stage != 0 {
                splice_ops_len += 1;
            }
            if op_id == jam_loop_id {
                jam_loop_i = i;
                stage = 1;
            } else if op_id == inner_loop_id {
                stage = 2;
            }
            match stage {
                0 => {}
                1 => {
                    pre_loop_ops.push(op_id);
                }
                2 => {
                    inner_loop_ops.push(op_id);
                    match self.ops[op_id] {
                        Op::Loop { .. } => {
                            inner_loop_level += 1;
                        }
                        Op::EndLoop => {
                            inner_loop_level -= 1;
                            if inner_loop_level == 0 {
                                stage = 3;
                            }
                        }
                        _ => {}
                    }
                }
                3 => {
                    post_loop_ops.push(op_id);
                    if self.ops[op_id] == Op::EndLoop {
                        break;
                    }
                }
                _ => {}
            }
        }
        println!("pre_loop_ops {pre_loop_ops:?}\ninner_loop_ops {inner_loop_ops:?}\npost_loop_ops {post_loop_ops:?}");
        println!("splice_ops_len {splice_ops_len}");

        debug_assert!(!pre_loop_ops.is_empty());
        debug_assert!(!inner_loop_ops.is_empty());
        debug_assert!(!post_loop_ops.is_empty());

        let Op::Loop { dim: jam_dim, scope } = self.ops[jam_loop_id] else { unreachable!() };
        debug_assert_eq!(scope, Scope::Register);

        let mut order = Vec::with_capacity(inner_loop_ops.len() + pre_loop_ops.len() * 2 + 5);
        let const_dim = self.ops.push(Op::Const(Constant::idx(jam_dim as u64)));
        order.push(const_dim);

        // Pre loop
        let mut defines = Set::default();
        for &op_id in &pre_loop_ops {
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id] {
                self.ops[op_id] = Op::Define { dtype, scope, ro, len: len * jam_dim };
                order.push(op_id);
                defines.insert(op_id);
            }
        }
        for &op_id in &pre_loop_ops {
            match self.ops[op_id] {
                //Op::Load { .. } => unreachable!(), // from the apply_optimization op this is invariant
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.ops.push(Op::Binary { x: index, y: const_dim, bop: BOp::Mul });
                        order.push(x);
                        let new_index = self.ops.push(Op::Binary { x, y: jam_loop_id, bop: BOp::Add });
                        order.push(new_index);
                        let Op::Store { index, .. } = &mut self.ops[op_id] else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Define { .. } => continue,
                _ => {}
            }
            order.push(op_id);
        }
        order.push(self.ops.push(Op::EndLoop));

        // Inner loop
        let mut remapping = Map::default();
        order.push(inner_loop_ops.remove(0)); // first is the Op::Loop
        for &op_id in &pre_loop_ops {
            let mut op = self.ops[op_id].clone();
            if !matches!(op, Op::Define { .. } | Op::Store { .. } | Op::Load { .. }) {
                op.remap_params(&remapping);
                let t_op_id = self.ops.push(op);
                order.push(t_op_id);
                remapping.insert(op_id, t_op_id);
            }
        }
        for &op_id in &inner_loop_ops {
            self.ops[op_id].remap_params(&remapping);
            match self.ops[op_id] {
                Op::Load { src, index } => {
                    if defines.contains(&src) {
                        let x = self.ops.push(Op::Binary { x: index, y: const_dim, bop: BOp::Mul });
                        order.push(x);
                        let new_index = self.ops.push(Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        order.push(new_index);
                        let Op::Load { index, .. } = &mut self.ops[op_id] else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.ops.push(Op::Binary { x: index, y: const_dim, bop: BOp::Mul });
                        order.push(x);
                        let new_index = self.ops.push(Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        order.push(new_index);
                        let Op::Store { index, .. } = &mut self.ops[op_id] else { unreachable!() };
                        *index = new_index;
                    }
                }
                _ => {}
            }
            order.push(op_id);
        }
        order.push(self.ops.push(Op::EndLoop));

        // Post Loop
        for &op_id in &pre_loop_ops {
            let mut op = self.ops[op_id].clone();
            if !matches!(op, Op::Define { .. } | Op::Store { .. } | Op::Load { .. }) {
                op.remap_params(&remapping);
                let t_op_id = self.ops.push(op);
                order.push(t_op_id);
                remapping.insert(op_id, t_op_id);
            }
        }
        for &op_id in &post_loop_ops {
            self.ops[op_id].remap_params(&remapping);
            match self.ops[op_id] {
                Op::Load { src, index } => {
                    if defines.contains(&src) {
                        let x = self.ops.push(Op::Binary { x: index, y: const_dim, bop: BOp::Mul });
                        order.push(x);
                        let new_index = self.ops.push(Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        order.push(new_index);
                        let Op::Load { index, .. } = &mut self.ops[op_id] else { unreachable!() };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.ops.push(Op::Binary { x: index, y: const_dim, bop: BOp::Mul });
                        order.push(x);
                        let new_index = self.ops.push(Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        order.push(new_index);
                        let Op::Store { index, .. } = &mut self.ops[op_id] else { unreachable!() };
                        *index = new_index;
                    }
                }
                _ => {}
            }
            order.push(op_id);
        }

        self.order.splice(jam_loop_i..jam_loop_i + splice_ops_len + 1, order);

        #[cfg(debug_assertions)]
        self.verify();
    }
}
