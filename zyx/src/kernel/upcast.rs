// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(clippy::while_let_loop)]

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

// ## Coalesced local+upcast access
//
// After `opt_split_global_to_local` and `upcast` splits axis 0 (M) and `opt_upcast`
// vectorizes, the address expression for the fine M-dimension is:
//
// ```text
// addr = gidx0 * TILE + lidx0 * V + i      (i = 0..V-1)
// ```
//
// where `lidx0 = threadIdx.x` and `V` is the vector width (e.g. 8).
//
// **Problem:** Consecutive threads (lidx0 = 0, 1, 2, ...) access elements
// spaced `V` apart (stride V).  A warp of 32 threads at stride 8 needs
// ~32× more cache line transactions than stride 1.
//
// **Fix:** Swap the multiplier `V` from `lidx0` to the upcast constant `i`:
//
// ```text
// addr = gidx0 * TILE + lidx0 + i * V
// ```
//
// Now consecutive threads access consecutive elements (stride 1) while
// each thread still covers `V` elements (spaced `V` apart within each
// thread's own access stream).  Per-step coalescing goes from 1 util/32 B
// to 8 utils/32 B.
//
// ### IR pattern
//
// The expression tree is nested — the `Mul(lidx, V)` and `Const(i)` are
// typically at different depths:
//
// ```text
// Add(Add(Mul(gidx, TILE), Mul(lidx, V)), Const(i))
// ```
//
// This pass searches for any `Add` where one subtree contains `Mul(lidx, V)`
// and the other is `Const(i)` with `i < V`.  The `Mul(lidx, V)` subtree is
// then rewritten to just `lidx`, and `Const(i)` becomes `Mul(Const(i), V)`.

impl Kernel {
    pub fn opt_upcast(&self) -> (Optimization, usize) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_upcast");
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
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("upcast");
        let Op::Index { len, scope, axis } = self.ops[gidx_id].op else { unreachable!() };
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

        if self.ops.len().0 as u64 * factor > 10000 {
            return;
        }

        // First skip ops that don't need duplication
        let mut op_id = self.head;
        while !op_id.is_null()
            && matches!(
                self.ops[op_id].op,
                Op::Define { scope: Scope::Global | Scope::Local, .. } | Op::Index { .. } | Op::Const(_)
            )
        {
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
        for &offset in &offsets {
            id = self.insert_after(id, Op::Binary { x: gidx_id, y: offset, bop: BOp::Add });
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
                Op::Index { .. } | Op::Loop { .. } | Op::EndLoop | Op::If { .. } | Op::EndIf | Op::Barrier { .. } => {}
                Op::Store { dst, x, index, vlen } => {
                    let mut ids = Vec::with_capacity((factor - 1) as usize);
                    let mut id = op_id;
                    if acc_defines.contains(&dst) {
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
                    } else {
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
                    }
                    remaps.insert(op_id, ids);
                }
                Op::Load { src, index, vlen } => {
                    let mut ids = Vec::with_capacity((factor - 1) as usize);
                    let mut id = op_id;
                    if acc_defines.contains(&src) {
                        for &offset in &offsets {
                            let index = self.insert_before(id, Op::Mad { x: index, y: const_factor, z: offset });
                            id = self.insert_after(index, Op::Load { src, index, vlen });
                            ids.push(id);
                        }
                        let index = self.insert_before(op_id, Op::Binary { x: index, y: const_factor, bop: BOp::Mul });
                        self.ops[op_id].op = Op::Load { src, index, vlen };
                    } else {
                        for i in 0..(factor - 1) as usize {
                            let mut index = index;
                            if let Some(remap) = remaps.get(&index) {
                                index = remap[i];
                            }
                            id = self.insert_after(id, Op::Load { src, index, vlen });
                            ids.push(id);
                        }
                    }
                    remaps.insert(op_id, ids);
                }
                ref op => {
                    let op = op.clone();
                    let mut ids = Vec::with_capacity((factor - 1) as usize);
                    let mut id = op_id;
                    for i in 0..(factor - 1) as usize {
                        let mut op = op.clone();
                        // Reindex the op
                        for param in op.parameters_mut() {
                            if let Some(remap) = remaps.get(param) {
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
