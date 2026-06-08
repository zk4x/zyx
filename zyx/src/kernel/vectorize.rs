// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Map,
    kernel::{Kernel, MemLayout, Op, OpId},
};

#[derive(Debug)]
struct LoadInfo {
    id: OpId,
    index: OpId,
}

impl Kernel {
    #[allow(unused)]
    pub fn vectorize_loads(&mut self) {
        // TODO for now this function ignores aliasing of stores and loads.
        // So later we need to make sure there are no aliasing issues

        let mut op_id = self.head;
        // Map: src id -> LoadInfo
        let mut loads: Vec<Map<OpId, Vec<LoadInfo>>> = Vec::new();
        loads.push(Map::default());
        'a: while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loads.push(Map::default());
                }
                Op::Load { src, index, layout } => {
                    if layout == MemLayout::Scalar {
                        loads
                            .last_mut()
                            .unwrap()
                            .entry(src)
                            .and_modify(|e| e.push(LoadInfo { id: op_id, index }))
                            .or_insert_with(|| vec![LoadInfo { id: op_id, index }]);
                    }
                }
                Op::EndLoop => self.fun_name(&mut loads),
                _ => {}
            }

            op_id = self.next_op(op_id);
        }

        self.fun_name(&mut loads);

        self.debug();

        todo!();
    }

    fn fun_name(&mut self, loads: &mut Vec<Map<OpId, Vec<LoadInfo>>>) {
        if let Some(mut loads) = loads.pop() {
            for (src, loads) in loads {
                for load in &loads {
                    println!("load {load:?}");
                    println!("{:?}", self.get_strides(load.index));
                }
                /*let mut i = 0;
                for load in &loads {
                    if load.offset != i {
                        return;
                    }
                    i += 1;
                }*/

                // Get the base index

                // Now that we know offsets are continues, we can replace the loads with single vectorized load
                /*if base_index == OpId::NULL {
                    base_index = self.insert_before(loads[0].id, Op::Const(Constant::idx(0)));
                }
                let vload = self.insert_before(
                    loads[0].id,
                    Op::Load { src: loads[0].src, index: base_index, layout: MemLayout::Vector(loads.len().try_into().unwrap()) },
                );
                for (idx, load) in loads.iter().enumerate() {
                    self.ops[load.id].op = Op::Devectorize { vec: vload, idx };
                }*/
            }
        }
    }

    #[allow(unused)]
    pub const fn vectorize_stores(_: &Kernel) {}
}
