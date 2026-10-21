// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use std::ops::RangeInclusive;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp},
    shape::Dim,
};

impl Kernel {
    /// Verify the kernel IR.
    ///
    /// Validates that the kernel has correct operation ordering
    /// (no uses before declarations) and proper data type propagation.
    /// This is an internal method used during kernel compilation.
    pub fn verify(&self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("verify");
        if !cfg!(debug_assertions) {
            return;
        }

        // Detect the kernel's linearization state from its stores — the rule
        // that always holds: a pre-linearize store writes a whole view to a
        // bare `Param`; linearization gives every store destination an
        // index expression. (Param shapes can NOT be used for detection:
        // `Variable` params have null shapes even pre-linearization.)
        let mut param_dst_stores = 0u32;
        let mut indexed_stores = 0u32;
        let mut has_post_linearize_ops = false;
        let mut has_move_or_reduce = false;
        {
            let mut scan = self.head;
            while !scan.is_null() {
                match self.at(scan) {
                    // No compile-time NaN may enter the IR: a folded NaN is
                    // always a bug (invalid folding or invalid input data).
                    Op::Const(c) => {
                        let is_nan = match c {
                            Constant::F32(x) => f32::from_le_bytes(*x).is_nan(),
                            Constant::F64(x) => f64::from_le_bytes(*x).is_nan(),
                            Constant::BF16(x) => u16::from_le_bytes(*x) & 0x7fff > 0x7f80,
                            Constant::F16(x) => {
                                let b = u16::from_le_bytes(*x);
                                b & 0x7c00 == 0x7c00 && b & 0x03ff != 0
                            }
                            _ => false,
                        };
                        if is_nan {
                            self.debug();
                        }
                        debug_assert!(!is_nan, "kernel contains a NaN constant at op {scan:?}");
                    }
                    Op::Store { dst, dst_idx, .. } => {
                        if dst_idx.is_null() && matches!(self.at(*dst), Op::Param { .. }) {
                            param_dst_stores += 1;
                        } else {
                            indexed_stores += 1;
                        }
                    }
                    // A traffic `Copy` is effect-only like a
                    // store: pure-copy kernels have no `Store`.
                    Op::Copy { .. } | Op::TT(TTOp::Copy { .. }) => {
                        indexed_stores += 1;
                    }
                    Op::Load { .. }
                    | Op::Storage { .. }
                    | Op::Range { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop
                    | Op::Index { .. }
                    | Op::Unroll { .. }
                    | Op::Barrier
                    | Op::Wmma { .. }
                    | Op::TT(TTOp::ReserveBack { .. })
                    | Op::TT(TTOp::PushBack { .. })
                    | Op::TT(TTOp::WaitFront { .. })
                    | Op::TT(TTOp::PopFront { .. })
                    | Op::TT(TTOp::MathLock)
                    | Op::TT(TTOp::MathUnlock)
                    | Op::TT(TTOp::PackLock)
                    | Op::TT(TTOp::PackUnlock)
                    | Op::TT(TTOp::NocReadBarrier)
                    | Op::TT(TTOp::NocWriteBarrier)
                    | Op::TT(TTOp::ReduceUninit)
                    | Op::TT(TTOp::EndReader)
                    | Op::TT(TTOp::EndCompute)
                    | Op::TT(TTOp::ProgramDesc(_))
                    | Op::TT(TTOp::TensixGrid(_))
                    | Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::FillTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::LLKMatmul { .. })
                    | Op::Asm { .. } => has_post_linearize_ops = true,
                    Op::Permute { .. }
                    | Op::Expand { .. }
                    | Op::Reshape { .. }
                    | Op::Pad { .. }
                    | Op::Narrow { .. }
                    | Op::Flip { .. } => has_move_or_reduce = true,
                    // Params and elementwise/compute ops appear in every
                    // kernel, pre- and post-linearization; they carry no
                    // linearization signal.
                    Op::Param { .. }
                    | Op::Cast { .. }
                    | Op::Bitcast { .. }
                    | Op::Unary { .. }
                    | Op::Binary { .. }
                    // Post-linearize lane-space reduces are legal until the
                    // device epilogue devectorizes them.
                    | Op::Reduce { .. }
                    | Op::Stack { .. } => {}
                    // Graph-only and post-render ops must never reach
                    // kernel IR at this (pre-render) stage.
                    Op::Source(_)
                    | Op::GPU(_)
                    | Op::Spirv(_)
                    | Op::PTX(_)
                    | Op::After { .. }
                    | Op::ToDevice { .. }
                    | Op::Contiguous { .. }
                    | Op::Program { .. }
                    | Op::Kernel(_) => {
                        todo!()
                    }
                }
                scan = self.next_op(scan);
            }
        }
        debug_assert!(param_dst_stores + indexed_stores > 0, "kernel must contain at least one store");
        if param_dst_stores > 0 && indexed_stores > 0 {
            println!("Invalid mixed kernel: whole-view stores and indexed stores/copies.");
            self.debug();
            panic!();
        }

        // Verify param/storage ordering: global params (RO) → GlobalMut params → local storages → everything else.
        // Only meaningful post-linearization; skipped for pre-linearize DAGs.
        if param_dst_stores == 0 {
            debug_assert!(!has_move_or_reduce, "post-linearize kernel must not contain Move/graph-shape ops");
            #[derive(PartialEq, Eq)]
            #[allow(dead_code)]
            enum Phase {
                GlobalRo,
                GlobalRw,
                LocalRo,
                LocalRw,
                Done,
            }
            let mut phase = Phase::GlobalRo;
            let mut scan = self.head;
            while !scan.is_null() {
                match self.at(scan) {
                    Op::Param { kind, .. } => match kind {
                        ParamKind::Variable | ParamKind::Global => {
                            if phase != Phase::GlobalRo {
                                println!("Global read-only params must come first.");
                                self.debug();
                                panic!();
                            }
                        }
                        ParamKind::GlobalMut => {
                            if phase == Phase::GlobalRo {
                                phase = Phase::GlobalRw;
                            }
                            if phase != Phase::GlobalRw {
                                println!("Global read-write params must come before local storages.");
                                self.debug();
                                panic!();
                            }
                        }
                    },
                    Op::Storage { scope: MemScope::Local, .. } | Op::Storage { scope: MemScope::Circular, .. } => {
                        // Relaxed: a Local storage may appear anywhere (e.g.
                        // custom kernels emit ranges/consts before planning
                        // smem tiles), so no ordering panic here.
                        if phase != Phase::LocalRw {
                            phase = Phase::LocalRw;
                        }
                    }
                    _ => {
                        if phase != Phase::Done {
                            phase = Phase::Done;
                        }
                    }
                }
                scan = self.next_op(scan);
            }
        } else {
            // Pre-linearize DAG: no lowered memory/control ops may exist.
            debug_assert!(!has_post_linearize_ops, "pre-linearize kernel must not contain Load/Copy/Storage/Loop ops");
        }

        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|set| set.contains(&x)) {
                println!("{op_id} {:?} uses {x} -> {:?} before declaration.", self.ops[op_id].op, self.ops[x].op);
                self.debug();
                panic!();
            }
        };

        let mut gids = Set::default();
        let mut lids = Set::default();

        let mut params: Map<OpId, ParamKind> = Map::default();
        let mut storages: Map<OpId, (MemScope, Dim)> = Map::default();

        let mut op_id = self.head;
        let mut prev: OpId;
        let mut dtypes: Map<OpId, DType> = Map::default();
        // Value shapes for the access checks below: single map
        // computation up front, never `Kernel::shape` (which is a
        // check-free accessor, not a verifier).
        let (shapes, _) = self.compute_dtypes_and_rcs();
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Store { dst, dst_idx, src: x } => {
                    check(op_id, dst, &stack);
                    check(op_id, x, &stack);
                    match self.at(dst) {
                        // Pre-linearize whole-view write: no index.
                        Op::Param { .. } => {
                            if dst_idx.is_null() {
                                // Pre-linearize write to a bare Param,
                                // whole view: no index.
                                // Post-linearize scalar checks now.
                            } else {
                                // Post-linearize element write into a
                                // global-mut Param: index required.
                                debug_assert_eq!(dtypes[&dst_idx], IDX_T, "store index must be {IDX_T}");
                                check(op_id, dst_idx, &stack);
                            }
                        }
                        Op::Storage { .. } => {
                            if !params.contains_key(&dst) && !storages.contains_key(&dst) {
                                println!("store={op_id} targets unbound variable");
                                self.debug();
                                panic!();
                            }
                            check(op_id, dst, &stack);
                            // A store's access shape (its index expres-
                            // sion's lane shape) must match its src
                            // value's shape: a tile store carries a tile
                            // access and its src must be a tile value (a
                            // scalar stored as a tile is a builder
                            // misuse, not a valid store). A src whose
                            // lanes are a subset of the access lanes is a
                            // broadcast (one value written to every lane
                            // address sharing its lanes): every unroll
                            // node in the src cone must appear in the
                            // index cone, and (both dim lists sorted
                            // ascending by axis) the src dims must form a
                            // subsequence of the access dims.
                            let src_shape = shapes[&x].1.clone();
                            if !dst_idx.is_null() {
                                let access_shape = shapes[&dst_idx].1.clone();
                                // Scalar src broadcasts to any access.
                                let mut subset = src_shape == access_shape || src_shape.is_empty();
                                if !subset && !src_shape.is_empty() {
                                    let mut src_seen = Set::default();
                                    let mut src_unrolls = Set::default();
                                    let mut stack = vec![x];
                                    while let Some(id) = stack.pop() {
                                        if !src_seen.insert(id) {
                                            continue;
                                        }
                                        if matches!(self.ops[id].op, Op::Unroll { .. }) {
                                            src_unrolls.insert(id);
                                        }
                                        stack.extend(self.ops[id].op.parameters());
                                    }
                                    let mut idx_seen = Set::default();
                                    let mut idx_unrolls = Set::default();
                                    stack = vec![dst_idx];
                                    while let Some(id) = stack.pop() {
                                        if !idx_seen.insert(id) {
                                            continue;
                                        }
                                        if matches!(self.ops[id].op, Op::Unroll { .. }) {
                                            idx_unrolls.insert(id);
                                        }
                                        stack.extend(self.ops[id].op.parameters());
                                    }
                                    let mut rest = access_shape.iter();
                                    subset = src_unrolls.iter().all(|u| idx_unrolls.contains(u))
                                        && src_shape.iter().all(|d| rest.any(|e| e == d));
                                }
                                if !subset {
                                    println!("store={op_id} access index {dst_idx:?} does not match src {x} shape {src_shape:?}");
                                    self.debug();
                                    panic!();
                                }
                            }
                            debug_assert!(!dst_idx.is_null(), "storage store without index");
                            debug_assert_eq!(dtypes[&dst_idx], IDX_T, "store index must be {IDX_T}");
                            check(op_id, dst_idx, &stack);
                        }
                        _ => {
                            println!("store={op_id} dst is not a Param or Storage");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Cast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Bitcast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Reduce { x, reduce_axis, .. } => {
                    // Lane-space reduce: a value op whose operands are the
                    // reduced value and the axis expression naming the
                    // reduced lane space.
                    check(op_id, x, &stack);
                    check(op_id, reduce_axis, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::TT(TTOp::ReserveBack { cb, n })
                | Op::TT(TTOp::PushBack { cb, n })
                | Op::TT(TTOp::WaitFront { cb, n })
                | Op::TT(TTOp::PopFront { cb, n }) => {
                    // Whole-buffer CB sync: `cb` names the Circular
                    // storage directly (no index — a reserve/wait/push/pop
                    // addresses no single location).
                    check(op_id, cb, &stack);
                    if !matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }) {
                        println!("cb sync op={op_id} targets non-circular {cb}");
                        self.debug();
                        panic!();
                    }
                    debug_assert!(n != 0, "cb sync op={op_id} has zero count");
                    dtypes.insert(op_id, dtypes[&cb]);
                }
                Op::TT(TTOp::LLK { ref ops, .. }) => {
                    // Opaque LLK call: operands are bare storages
                    // (Circular CBs plus the Register acc slot) and
                    // value ops (index arithmetic). Per-position scope
                    // enforcement is the caller's job (tt_storage's
                    // debug_assert!); verify only rejects operands that
                    // are neither a storage nor a value. NULL operands
                    // are render-filled placeholders (the result slot)
                    // — not scope-checked, never indexed.
                    for &x in ops.iter() {
                        if x.is_null() {
                            continue;
                        }
                        check(op_id, x, &stack);
                        if !matches!(
                            self.at(x),
                            Op::Storage { .. }
                                | Op::Const(_)
                                | Op::Unroll { .. }
                                | Op::Param { .. }
                                | Op::Cast { .. }
                                | Op::Bitcast { .. }
                                | Op::Unary { .. }
                                | Op::Binary { .. }
                                | Op::Stack { .. }
                                | Op::Load { .. }
                                | Op::Copy { .. }
                                | Op::Range { .. }
                                | Op::Loop { .. }
                                | Op::Index { .. }
                        ) {
                            println!("tt llk op={op_id} has non-storage/non-value operand {x}");
                            self.debug();
                            panic!();
                        }
                    }
                    if !ops.is_empty() {
                        dtypes.insert(op_id, dtypes[&ops[0]]);
                    }
                }
                Op::TT(TTOp::ReduceTile { cb_in, cb_sc, slot, .. }) => {
                    // Lowered reduce: bare storages only (input/scaler
                    // CBs, acc slot). Result takes the input CB's dtype.
                    for &f in &[cb_in, cb_sc, slot] {
                        check(op_id, f, &stack);
                        if !matches!(self.at(f), Op::Storage { .. }) {
                            println!("tt reduce tile op={op_id} has non-storage operand {f}");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtypes[&cb_in]);
                }
                Op::TT(TTOp::FillTile { slot, val }) => {
                    // Lowered fill: Register slot plus the scalar Const.
                    // Result is the fill value broadcast to one tile.
                    for &f in &[slot, val] {
                        check(op_id, f, &stack);
                        if !matches!(
                            self.at(f),
                            Op::Storage { scope: MemScope::Register, .. } | Op::Const(_)
                        ) {
                            println!("tt fill tile op={op_id} has non-slot/non-const operand {f}");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtypes[&val]);
                }
                Op::TT(TTOp::LLKMatmul { cb_a, cb_b, slot }) => {
                    // Lowered matmul: bare storages only (input CBs,
                    // result slot). Result takes the left CB's dtype.
                    for &f in &[cb_a, cb_b, slot] {
                        check(op_id, f, &stack);
                        if !matches!(self.at(f), Op::Storage { .. }) {
                            println!("tt matmul op={op_id} has non-storage operand {f}");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtypes[&cb_a]);
                }
                Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes }) => {
                    // Lowered traffic copy: DRAM param or Circular
                    // storage sides, unroll-free scalar indices,
                    // explicit non-zero size. No value produced.
                    for &f in &[src, src_idx, dst, dst_idx] {
                        check(op_id, f, &stack);
                    }
                    if !matches!(
                        self.at(src),
                        Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }
                            | Op::Storage { scope: MemScope::Circular, .. }
                    ) {
                        println!("tt copy op={op_id} has non-traffic src {src:?}");
                        self.debug();
                        panic!();
                    }
                    if !matches!(
                        self.at(dst),
                        Op::Param { kind: ParamKind::GlobalMut, .. } | Op::Storage { scope: MemScope::Circular, .. }
                    ) {
                        println!("tt copy op={op_id} has non-traffic dst {dst:?}");
                        self.debug();
                        panic!();
                    }
                    for &i in &[src_idx, dst_idx] {
                        if self.contains_unroll(i) {
                            println!("tt copy op={op_id} indexes with lanes at {i:?}");
                            self.debug();
                            panic!();
                        }
                    }
                    if bytes == 0 {
                        println!("tt copy op={op_id} has zero size");
                        self.debug();
                        panic!();
                    }
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    // Lowered fused broadcast: same rule over cb_a.
                    for &f in &[cb_a, cb_b, mx, plain] {
                        check(op_id, f, &stack);
                        if !matches!(
                            self.at(f),
                            Op::Storage { .. }
                                | Op::Const(_)
                                | Op::Unroll { .. }
                                | Op::Param { .. }
                                | Op::Cast { .. }
                                | Op::Bitcast { .. }
                                | Op::Unary { .. }
                                | Op::Binary { .. }
                                | Op::Stack { .. }
                                | Op::Load { .. }
                                | Op::Copy { .. }
                                | Op::Range { .. }
                                | Op::Loop { .. }
                                | Op::Index { .. }
                        ) {
                            println!("tt llk bcast op={op_id} has non-storage/non-value operand {f}");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtypes[&cb_a]);
                }
                Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::ProgramDesc(_))
                | Op::TT(TTOp::TensixGrid(_)) => {}
                Op::Unary { x, .. }
                | Op::Permute { x, .. }
                | Op::Pad { x, .. }
                | Op::Expand { x, .. }
                | Op::Reshape { x, .. }
                | Op::Flip { x, .. }
                | Op::Narrow { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Binary { x, y, bop } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    // C-like promotion: F16 x F32 -> F32. A tile matmul is
                    // typed by its inputs (F16) but holds F32 under 32-bit
                    // DST, and the emitted add_binary_tile is F32-correct.
                    let dtype = match (dtypes[&x], dtypes[&y]) {
                        (DType::F16, DType::F32) | (DType::F32, DType::F16) => DType::F32,
                        (dx, dy) if dx == dy => dx,
                        _ => {
                            println!("Binary dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    };
                    if bop.returns_bool() {
                        dtypes.insert(op_id, DType::Bool);
                    } else {
                        dtypes.insert(op_id, dtype);
                    }
                    // A binary's dtype compatibility is checked above;
                    // shape compatibility is the passes' own business
                    // (symbolic checks happen at tensor level, never in
                    // the kernel), so no shape proof is forced here.
                }
                Op::Asm { ref ops, .. } => {
                    if ops.is_empty() {
                        // Operand-free Asm is an effect-only call (e.g. a
                        // TT LLK init): nothing to check, no value.
                    } else {
                        let dtype = dtypes[&ops[0]];
                        for &x in ops.iter() {
                            check(op_id, x, &stack);
                        }
                        // Asm may mix dtypes (e.g. U32 qs, I64 intra, F16 scale/min) — like CUDA C mixed arithmetic.
                        // Result dtype is ops[0]'s dtype (e.g. F16 scale), no cross-check.
                        dtypes.insert(op_id, dtype);
                    }
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops.iter() {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            println!("Vectorize dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Index { vec, .. } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                }
                Op::Wmma { c, a, b, .. } => {
                    let dtype = dtypes[&c];
                    check(op_id, c, &stack);
                    check(op_id, a, &stack);
                    check(op_id, b, &stack);
                    if dtypes[&a] != dtypes[&b] {
                        println!("MMA dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Const(v) => {
                    dtypes.insert(op_id, v.dtype());
                }
                Op::Unroll { len, .. } => {
                    dtypes.insert(op_id, IDX_T);
                    debug_assert!(len > 0, "unroll with zero lanes");
                }
                Op::Param { dtype, kind, shape, .. } => {
                    params.insert(op_id, kind);
                    dtypes.insert(op_id, dtype);
                    if shape != OpId::NULL {
                        check(op_id, shape, &stack);
                    }
                }
                Op::Storage { dtype, scope, len } => {
                    storages.insert(op_id, (scope, len));
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, src_idx: index } => {
                    if !params.contains_key(&src) && !storages.contains_key(&src) {
                        println!("load={op_id} is trying to load from unbound variable");
                        self.debug();
                        panic!();
                    }
                    debug_assert_eq!(dtypes[&index], IDX_T, "load index must be {IDX_T}");
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Copy { src, src_idx, dst, dst_idx } => {
                    for &side in &[src, dst] {
                        if !matches!(self.at(side), Op::Param { .. } | Op::Storage { .. }) {
                            println!("copy={op_id} side is not a Param or Storage");
                            self.debug();
                            panic!();
                        }
                        check(op_id, side, &stack);
                    }
                    for &o in &[src_idx, dst_idx] {
                        if !o.is_null() {
                            check(op_id, o, &stack);
                        }
                    }
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Range { axis, kind: scope, .. } => {
                    match scope {
                        RangeKind::Group(len) => {
                            if !gids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                            if let Some(d) = self.resolve_const(len).and_then(Constant::as_dim)
                                && d < 0
                            {
                                println!("Group index length resolves to negative constant {d} at op {op_id:?}");
                                self.debug();
                                panic!();
                            }
                            check(op_id, len, &stack);
                        }
                        RangeKind::Local(_) => {
                            if !lids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        RangeKind::Warp(local_id) => {
                            // A warp is a view over a local range on the same axis: the
                            // local range owns the axis, so only the reference is validated.
                            match self.ops[local_id].op {
                                Op::Range { axis: ref_axis, kind: RangeKind::Local(_) } if ref_axis == axis => {}
                                _ => {
                                    println!(
                                        "index={op_id} warp references op {local_id}, which is not a local range on axis {axis}"
                                    );
                                    self.debug();
                                    panic!();
                                }
                            }
                        }
                    }
                    dtypes.insert(op_id, IDX_T);
                }
                Op::Loop { len } => {
                    // Boolean length = conditional (0-or-1 trip): the old
                    // `If` boolean assertion lives here now. Counted loops
                    // keep the negativity check.
                    if self.dtype(len) != DType::Bool
                        && let Some(d) = self.resolve_const(len).and_then(Constant::as_dim)
                        && d < 0
                    {
                        println!("Loop length resolves to negative constant {d} at op {op_id:?}");
                        self.debug();
                        panic!();
                    }
                    check(op_id, len, &stack);
                    stack.push(Set::default());
                    dtypes.insert(op_id, IDX_T);
                }
                Op::EndLoop => {
                    if stack.is_empty() {
                        println!("Endloop without matching loop.");
                        self.debug();
                        panic!();
                    }
                    stack.pop();
                }
                Op::Barrier => {}
                Op::Source(_)
                | Op::GPU(_)
                | Op::Spirv(_)
                | Op::PTX(_)
                | Op::After { .. }
                | Op::ToDevice { .. }
                | Op::Contiguous { .. }
                | Op::Program { .. }
                | Op::Kernel(_) => {
                    todo!()
                }
            }
            stack.last_mut().unwrap().insert(op_id);
            prev = op_id;
            op_id = self.ops[op_id].next;
            if !op_id.is_null() && self.ops[op_id].prev != prev {
                println!("Inconsistency in prev.");
                self.debug();
                panic!()
            }
        }
        if stack.len() != 1 {
            println!("Wrong {} closing endloops.", stack.len());
            self.debug();
            panic!();
        }
        self.check_oob();
    }

    pub(crate) fn check_oob(&self) {
        let mut storages = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Storage { len, .. } => {
                    storages.insert(op_id, len);
                }
                Op::Load { src, src_idx: index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range
                        && let Some(&len) = storages.get(&src)
                        && *range.end() >= len
                    {
                        self.debug();
                        panic!("OOB detected in op {}: index {:?} exceeds buffer length {:?}", op_id, range, len);
                    }
                }
                Op::Store { dst, dst_idx: index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range
                        && let Some(&len) = storages.get(&dst)
                        && *range.start() > len + 1
                    {
                        self.debug();
                        panic!("OOB detected in op {}: index {:?} exceeds buffer length {:?}", op_id, range, len);
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }
}

impl Kernel {
    /// Compute value-range bounds for every operation in the kernel.
    ///
    /// # Invariant
    ///
    /// `compute_bounds` can **never be precise**. It always returns
    /// **conservative (over-approximating)** bounds: for every op, its true
    /// runtime value is contained in `[lb, ub]`. The single guarantee we MUST
    /// uphold is that bounds are **never too tight** — they must never
    /// *under*-approximate the true range. Being wider than reality is always
    /// safe; being tighter than reality is the only forbidden failure mode,
    /// because it would let a constant fold assume a value the op can never take.
    ///
    /// # Why it is always imprecise
    ///
    /// The imprecision is fundamental and expected, not a bug: variables are
    /// bounded **independently**, which ignores *correlations* between them. For
    /// example, `x` and `y` may always satisfy `x <= y` at runtime, but their
    /// independent ranges are derived separately and will overlap / be wider
    /// than the true joint set of reachable values. The resulting range is
    /// therefore wider than reality — that is correct and intended. Precision
    /// can never be recovered without tracking joint constraints, which this
    /// pass deliberately does not do.
    ///
    /// Because the bounds never under-approximate, they are safe to use for
    /// proving a comparison or boolean op is statically constant (if the
    /// conservative range already forces the comparison to one result, that
    /// result holds for every concrete value). They are **NOT** safe for
    /// replacing an op with a specific non-constant value, only for deciding
    /// constant outcomes.
    #[allow(clippy::match_same_arms)]
    pub(crate) fn compute_bounds(&self) -> Map<OpId, (Dim, Dim)> {
        // Single linear walk, O(number of ops). Bounds are ALWAYS conservative
        // (wide): we never narrow from guard conditions and never narrow across
        // scopes, so a single global map suffices — no scope stack, no cloning,
        // no per-op merge. Each op's bound is derived once from its
        // (already-processed) operands. This is intentionally not precise (see
        // the doc comment above): variables are bounded independently and
        // correlations are ignored, so ranges are wider than reality, which is
        // correct and required.
        let mut bounds: Map<OpId, (Dim, Dim)> = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    if let Some(v) = x.as_dim() {
                        bounds.insert(op_id, (v, v));
                    }
                }
                Op::Storage { .. } => {}
                Op::Loop { .. } | Op::Unary { .. } | Op::Cast { .. } | Op::Binary { .. } => {
                    self.rederive_bounds(&mut bounds, op_id);
                }
                Op::Range { kind: scope, .. } => {
                    let len = match scope {
                        RangeKind::Group(len) => self.resolve_const(len).and_then(crate::dtype::Constant::as_dim),
                        RangeKind::Local(len) => Some(i64::from(len)),
                        // A warp's value is the lane id: bounded by the warp size.
                        RangeKind::Warp(_) => Some(i64::from(self.dev_info().warp_size)),
                    };
                    // An unresolved (dynamic) group length is UNKNOWN: no bounds
                    // must be fabricated for it. A huge sentinel here would wrap
                    // around in downstream arithmetic and produce false tight
                    // ranges (-> provably-false guards -> wrong constant folding).
                    if let Some(len) = len {
                        bounds.insert(op_id, (0, len.saturating_sub(1)));
                    }
                }
                Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                    let mut r = None;
                    for x in ops.iter() {
                        if let Some(&(xl, xu)) = bounds.get(x) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::TT(TTOp::ReduceTile { cb_in, cb_sc, slot, .. }) => {
                    let mut r = None;
                    for f in [cb_in, cb_sc, slot] {
                        if let Some(&(xl, xu)) = bounds.get(&f) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::TT(TTOp::FillTile { slot, val }) => {
                    let mut r = None;
                    for f in [slot, val] {
                        if let Some(&(xl, xu)) = bounds.get(&f) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::TT(TTOp::LLKMatmul { cb_a, cb_b, slot }) => {
                    let mut r = None;
                    for f in [cb_a, cb_b, slot] {
                        if let Some(&(xl, xu)) = bounds.get(&f) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    let mut r = None;
                    for f in [cb_a, cb_b, mx, plain] {
                        if let Some(&(xl, xu)) = bounds.get(&f) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::Stack { ref ops } => {
                    let mut r = None;
                    for x in ops.iter() {
                        if let Some(&(xl, xu)) = bounds.get(x) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
        bounds
    }

    fn rederive_bounds(&self, prev: &mut Map<OpId, (Dim, Dim)>, op_id: OpId) {
        match *self.at(op_id) {
            Op::Cast { x, .. } => {
                if let Some(&b) = prev.get(&x) {
                    prev.insert(op_id, b);
                }
            }
            Op::Binary { x, y, bop } => {
                let Some(&(min_x, max_x)) = prev.get(&x) else { return };
                let Some(&(min_y, max_y)) = prev.get(&y) else { return };
                let range = match bop {
                    // Saturating, never wrapping: an overflow must not
                    // fabricate a small upper bound out of huge ones.
                    BOp::Add => (min_x.saturating_add(min_y), max_x.saturating_add(max_y)),
                    BOp::Sub => (min_x.saturating_sub(max_y), max_x.saturating_sub(min_y)),
                    BOp::Mul => {
                        // The true range is the min/max over the four corner
                        // products; the naive (min_x*min_y, max_x*max_y) is only
                        // valid for non-negative operands and under-approximates
                        // (non-conservative) when signs mix.
                        let p1 = min_x.saturating_mul(min_y);
                        let p2 = min_x.saturating_mul(max_y);
                        let p3 = max_x.saturating_mul(min_y);
                        let p4 = max_x.saturating_mul(max_y);
                        (p1.min(p2).min(p3).min(p4), p1.max(p2).max(p3).max(p4))
                    }
                    BOp::Div | BOp::Mod if min_y == 0 || max_y == 0 => (Dim::MIN, Dim::MAX),
                    BOp::Div => {
                        // x / y over the rectangle: min/max of the four corner
                        // quotients (saturating — a divisor near zero would
                        // otherwise fabricate a tiny bound).
                        let q1 = min_x.saturating_div(min_y);
                        let q2 = min_x.saturating_div(max_y);
                        let q3 = max_x.saturating_div(min_y);
                        let q4 = max_x.saturating_div(max_y);
                        (q1.min(q2).min(q3).min(q4), q1.max(q2).max(q3).max(q4))
                    }
                    BOp::Mod => {
                        // zyx integer remainder has the same sign as the
                        // dividend (truncated division), so `|x % y| < |y|` and
                        // the sign follows `x`. When the dividend is known
                        // non-negative the remainder lies in `[0, |y|-1]`, and
                        // when it is known non-positive in `[-(|y|-1), 0]`.
                        // These are sound (conservative) tightenings of the
                        // sign-agnostic `[-(|y|-1), |y|-1]`.
                        let mag = max_y.unsigned_abs().max(min_y.unsigned_abs());
                        if mag == 0 {
                            (Dim::MIN, Dim::MAX)
                        } else {
                            let m = mag as i64 - 1;
                            if min_x >= 0 {
                                (0, m)
                            } else if max_x <= 0 {
                                (-m, 0)
                            } else {
                                (-m, m)
                            }
                        }
                    }
                    BOp::BitShiftLeft => (min_x << min_y.min(63), max_x << max_y.min(63)),
                    BOp::BitShiftRight => (min_x >> min_y.min(63), max_x >> max_y.min(63)),
                    BOp::Pow => {
                        let min_val = if min_y == 0 {
                            1
                        } else if min_x == 0 {
                            0
                        } else {
                            min_x.saturating_pow(min_y.min(u32::MAX as i64) as u32)
                        };
                        let max_val = if max_y == 0 {
                            1
                        } else if max_x == 0 {
                            0
                        } else {
                            max_x.saturating_pow(max_y.min(u32::MAX as i64) as u32)
                        };
                        (min_val, max_val)
                    }
                    BOp::Eq => {
                        let always = (min_x == max_x) && (min_y == max_y) && (min_x == min_y);
                        let maybe = !(max_x < min_y || max_y < min_x || always);
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::NotEq => {
                        let always = max_x < min_y || max_y < min_x;
                        let maybe = !(always || min_x == max_x && min_y == max_y && min_x == min_y);
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmpgt => {
                        let always = min_x > max_y;
                        let never = max_x <= min_y;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmpge => {
                        let always = min_x >= max_y;
                        let never = max_x < min_y;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmplt => {
                        let always = max_x < min_y;
                        let never = max_y <= min_x;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::And => {
                        let always = (min_x == 1 && max_x == 1) && (min_y == 1 && max_y == 1);
                        let maybe = (max_x >= 1) && (max_y >= 1);
                        (Dim::from(always as u8), Dim::from((always || maybe) as u8))
                    }
                    BOp::Or => {
                        let always = (min_x == 1 && max_x == 1) || (min_y == 1 && max_y == 1);
                        let maybe = (min_x == 1) || (min_y == 1) || (max_x == 1) || (max_y == 1);
                        (Dim::from(always), Dim::from(always || maybe))
                    }
                    BOp::Max => (min_x.max(min_y), max_x.max(max_y)),
                    BOp::BitAnd => (0, max_x.min(max_y)),
                    BOp::BitOr => (min_x | min_y, max_x | max_y),
                    BOp::BitXor => (0, max_x.max(max_y)),
                };
                prev.insert(op_id, range);
            }
            Op::Loop { len } => {
                if let Some(&(_, upper)) = prev.get(&len) {
                    prev.insert(op_id, (0, upper.saturating_sub(1)));
                }
            }
            _ => {}
        }
    }
}

impl Kernel {
    const fn get_bounds(_op_id: OpId) -> Option<RangeInclusive<Dim>> {
        // TODO
        None
    }
}
