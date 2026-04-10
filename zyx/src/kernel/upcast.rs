// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn opt_upcast(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global {
                    for f in [4, 8, 2, 16] {
                        let f = f as u64;
                        if len.is_multiple_of(f) && len / f >= 4 {
                            factors.push((op_id, f));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::Upcast { factors }, n_configs)
    }

    pub fn upcast(&mut self, gidx_id: OpId, factor: u64) {
        let Op::Index { len, scope, axis } = self.ops[gidx_id].op else {
            unreachable!()
        };
        debug_assert!(len.is_multiple_of(factor));
        debug_assert_eq!(scope, Scope::Global);

        //println!("upcast gidx_id={gidx_id} by factor={factor}");

        // === Some checks when we just cannot upcast === //

        // We cannot upcast if the kernel is already vectorized
        // Also let's not upcast kernel with barriers for now
        if self.ops.values().any(|node| match node.op {
            Op::Load { vlen, .. } | Op::Store { vlen, .. } => vlen != 1,
            Op::Barrier { .. } => true,
            _ => false,
        }) {
            return;
        }

        // First skip ops that don't need duplication
        let mut op_id = self.head;
        loop {
            match self.ops[op_id].op {
                Op::Define { scope: Scope::Global | Scope::Local, .. } => {}
                Op::Index { .. } => {}
                Op::Const(_) => {}
                _ => {
                    break;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Move gidx_id to the beginning, so that it's not getting transformed by it's own function
        self.move_op_before(gidx_id, op_id);

        // Create constant for factor
        let const_factor = self.insert_before(gidx_id, Op::Const(Constant::idx(factor as u64)));

        // Create index offsets
        let mut offsets = Vec::with_capacity((factor - 1) as usize);
        for i in 1..factor {
            offsets.push(self.insert_before(gidx_id, Op::Const(Constant::idx(i as u64))));
        }

        // For remapping parameters
        let mut remaps: Map<OpId, Vec<OpId>> = Map::default();

        // Global index now split into multiple indices with constant offsets
        let x = self.insert_before(gidx_id, Op::Index { len: len / factor, scope, axis });
        self.ops[gidx_id].op = Op::Binary { x, y: const_factor, bop: BOp::Mul };
        let mut ids = Vec::with_capacity((factor - 1) as usize);
        let mut id = gidx_id;
        for i in 0..(factor - 1) as usize {
            id = self.insert_after(id, Op::Binary { x: gidx_id, y: offsets[i], bop: BOp::Add });
            ids.push(id);
        }
        remaps.insert(gidx_id, ids);

        // Now loop over remaining ops and duplicate as needed
        let mut acc_defines = Set::default();
        while !op_id.is_null() {
            let next_op_id = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Define { dtype, scope: Scope::Register, ro, len } => {
                    self.ops[op_id].op = Op::Define { dtype, scope: Scope::Register, ro, len: len * factor };
                    acc_defines.insert(op_id);
                }
                Op::Loop { .. } => {}
                Op::EndLoop { .. } => {}
                Op::If { .. } => {}
                Op::EndIf => {}
                Op::Barrier { .. } => {}
                Op::Store { dst, x, index, vlen } => {
                    if acc_defines.contains(&dst) {
                        let mut ids = Vec::with_capacity((factor - 1) as usize);
                        let mut id = op_id;
                        for i in 0..(factor - 1) as usize {
                            let mut x = x;
                            if let Some(remap) = remaps.get(&x) {
                                x = remap[i];
                            }
                            let index = self.insert_before(id, Op::Mad { x: index, y: const_factor, z: offsets[i] });
                            id = self.insert_after(index, Op::Store { dst, x, index, vlen });
                            ids.push(id);
                        }
                        let index = self.insert_before(op_id, Op::Binary { x: index, y: const_factor, bop: BOp::Mul });
                        self.ops[op_id].op = Op::Store { dst, x, index, vlen };
                        remaps.insert(op_id, ids);
                    } else {
                        let mut ids = Vec::with_capacity((factor - 1) as usize);
                        let mut id = op_id;
                        for i in 0..(factor - 1) as usize {
                            let mut x = x;
                            if let Some(remap) = remaps.get(&x) {
                                x = remap[i];
                            }
                            let mut index = index;
                            if let Some(remap) = remaps.get(&index) {
                                index = remap[i];
                            }
                            id = self.insert_after(id, Op::Store { dst, x, index, vlen });
                            ids.push(id);
                        }
                        remaps.insert(op_id, ids);
                    }
                }
                Op::Load { src, index, vlen } => {
                    if acc_defines.contains(&src) {
                        let mut ids = Vec::with_capacity((factor - 1) as usize);
                        let mut id = op_id;
                        for i in 0..(factor - 1) as usize {
                            let index = self.insert_before(id, Op::Mad { x: index, y: const_factor, z: offsets[i] });
                            id = self.insert_after(index, Op::Load { src, index, vlen });
                            ids.push(id);
                        }
                        let index = self.insert_before(op_id, Op::Binary { x: index, y: const_factor, bop: BOp::Mul });
                        self.ops[op_id].op = Op::Load { src, index, vlen };
                        remaps.insert(op_id, ids);
                    } else {
                        let mut ids = Vec::with_capacity((factor - 1) as usize);
                        let mut id = op_id;
                        for i in 0..(factor - 1) as usize {
                            let mut index = index;
                            if let Some(remap) = remaps.get(&index) {
                                index = remap[i];
                            }
                            id = self.insert_after(id, Op::Load { src, index, vlen });
                            ids.push(id);
                        }
                        remaps.insert(op_id, ids);
                    }
                }
                ref op => {
                    let op = op.clone();
                    let mut ids = Vec::with_capacity((factor - 1) as usize);
                    let mut id = op_id;
                    for i in 0..(factor - 1) as usize {
                        let mut op = op.clone();
                        // Reindex the op
                        for param in op.parameters_mut() {
                            if let Some(remap) = remaps.get(&param) {
                                *param = remap[i];
                            }
                        }
                        id = self.insert_after(id, op);
                        ids.push(id);
                    }
                    // Store which remaps will be used in the future
                    remaps.insert(op_id, ids);
                }
            }
            op_id = next_op_id;
        }

        self.verify();
    }
}
