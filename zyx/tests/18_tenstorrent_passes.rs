// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent kernel passes, exercised as external pass authors would:
//! build IR with the public kernel API, run one pass, inspect the result
//! through `head`/`at`/`next_op`. No device needed.

use zyx::DType;
use zyx::kernel::{BOp, Dev, Kernel, MemScope, Op, OpId, TTOp, TileDim};

/// MATH tile-compute runs under `tile_regs_acquire..commit`;
/// the pack drain (Register slot stored into a Circular buffer)
/// runs under `tile_regs_wait..release`. Pure insertion: the
/// physical DST slots are the `MemScope::Register` storages
/// already in the IR, this pass assigns no numbers.
#[test]
fn lock_dst_wraps_math_and_pack() {
    let mut k = Kernel::new(Dev::Auto);
    let cb_a = k.storage(DType::F32, MemScope::Circular, 1024);
    let cb_b = k.storage(DType::F32, MemScope::Circular, 1024);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let cout = k.storage(DType::F32, MemScope::Circular, 1024);
    let zero = k.const_val(0i64);
    k.tt_end_reader();
    let va = k.load_circular(cb_a, zero);
    let vb = k.load_circular(cb_b, zero);
    let av = k.load_register_tile(acc, zero);
    let f = k.matmul_tile(va, vb, av);
    k.store_register_tile(acc, f, zero);
    let out = k.load_register_tile(acc, zero);
    k.store_circular(cout, out, zero);
    k.tt_end_compute();
    k.tt_lock_dst();

    let mut ops: Vec<Op> = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        ops.push(k.at(op_id).clone());
        op_id = k.next_op(op_id);
    }
    let math_locks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::MathLock))).count();
    let math_unlocks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::MathUnlock))).count();
    let pack_locks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::PackLock))).count();
    let pack_unlocks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::PackUnlock))).count();
    assert_eq!(math_locks, 1, "one MATH acquire per section with MATH ops");
    assert_eq!(math_unlocks, 1, "one MATH commit per section with MATH ops");
    assert_eq!(pack_locks, 1, "one pack wait per section with a pack drain");
    assert_eq!(pack_unlocks, 1, "one pack release per section with a pack drain");
    // Acquire precedes the matmul; release follows the store.
    let pos = |op: &Op| ops.iter().position(|o| o == op).unwrap();
    assert!(pos(&Op::TT(TTOp::MathLock)) < pos(&k.at(f).clone()));
    assert!(pos(&Op::TT(TTOp::PackUnlock)) > pos(&k.at(f).clone()));
}

/// `tt_sync_cbs` wraps CB traffic with straight-line single-tile
/// syncs: a publish copy gets reserve/push, a drain copy and a
/// compute-feeding circular load get wait-front. DRAM↔CB moves are
/// `Copy` ops; a circular load feeding anything but tile compute
/// panics.
#[test]
fn sync_cbs_wraps_cb_traffic() {
    let mut k = Kernel::new(Dev::Auto);
    let a = k.param(DType::F32);
    let out = k.param_mut(DType::F32);
    let ca = k.storage(DType::F32, MemScope::Circular, 1024);
    let cb = k.storage(DType::F32, MemScope::Circular, 1024);
    let cout = k.storage(DType::F32, MemScope::Circular, 1024);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0i64);
    let pub_a = k.copy_global_to_circular(a, zero, ca, zero);
    k.tt_end_reader();
    let va = k.load_circular(ca, zero);
    let vb = k.load_circular(cb, zero);
    let av = k.load_register_tile(acc, zero);
    let f = k.matmul_tile(va, vb, av);
    k.store_register_tile(acc, f, zero);
    k.tt_end_compute();
    let drain = k.copy_circular_to_global(cout, zero, out, zero);
    k.tt_sync_cbs();

    let mut order: Vec<OpId> = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        order.push(op_id);
        op_id = k.next_op(op_id);
    }
    let pos = |id: OpId| order.iter().position(|&o| o == id).unwrap();
    let find = |want: &Op| order.iter().find(|&&id| k.at(id) == want).copied().unwrap();
    let res = find(&Op::TT(TTOp::ReserveBack { cb: ca, n: 1 }));
    let push = find(&Op::TT(TTOp::PushBack { cb: ca, n: 1 }));
    assert_eq!(pos(res) + 1, pos(pub_a), "reserve sits immediately before the publish copy");
    assert_eq!(pos(push), pos(pub_a) + 1, "push sits immediately after the publish copy");
    let w = find(&Op::TT(TTOp::WaitFront { cb: ca, n: 1 }));
    assert_eq!(pos(w) + 1, pos(va), "wait sits immediately before the compute-feeding load");
    let wd = find(&Op::TT(TTOp::WaitFront { cb: cout, n: 1 }));
    assert_eq!(pos(wd) + 1, pos(drain), "wait sits immediately before the drain copy");
}

/// `tt_storage` rewrites the SSA tile-compute ops to opaque
/// `LLK` call templates over the bare `Storage`s already in
/// the IR. The matmul op becomes `matmul_tiles({0}, {1}, 0, 0, {2})`
/// over its two CB storages and the register slot; the reduce op
/// becomes `reduce_tile<{op},{dim}>({0}, {1}, 0, 0, {2})` over
/// input CB, scaler CB, register slot.
#[test]
fn storage_lowering_rewrites_matmul_and_reduce() {
    let mut k = Kernel::new(Dev::Auto);
    let cb_a = k.storage(DType::F32, MemScope::Circular, 1024);
    let cb_b = k.storage(DType::F32, MemScope::Circular, 1024);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let csc = k.storage(DType::F32, MemScope::Circular, 1024);
    let cout = k.storage(DType::F32, MemScope::Circular, 1024);
    let zero = k.const_val(0i64);
    k.tt_end_reader();
    let va = k.load_circular(cb_a, zero);
    let vb = k.load_circular(cb_b, zero);
    let av = k.load_register_tile(acc, zero);
    let f = k.matmul_tile(va, vb, av);
    let vs = k.load_circular(csc, zero);
    let r = k.reduce_tile(va, vs, av, BOp::Max, TileDim::Col);
    // Pack drain: the Register slot is read out and stored to a
    // CB. The matmul/reduce results are effect ops after
    // lowering (no SSA value to thread).
    let out = k.load_register_tile(acc, zero);
    k.store_circular(cout, out, zero);
    k.tt_end_compute();
    k.tt_storage();

    let matmul = k.at(f);
    let reduce = k.at(r);
    match matmul {
        Op::TT(TTOp::LLK { asm, ops }) => {
            assert_eq!(asm.as_str(), "matmul_tiles({0}, {1}, 0, 0, {2});");
            assert_eq!(ops.as_slice(), &[cb_a, cb_b, acc, va, vb]);
        }
        other => panic!("matmul not lowered to LLK, got {other:?}"),
    }
    match reduce {
        Op::TT(TTOp::LLKReduce { rop, kind, cb_in, cb_sc, slot, x, scaler }) => {
            assert!(matches!(rop, BOp::Max));
            assert!(matches!(kind, TileDim::Col));
            assert_eq!((*cb_in, *cb_sc, *slot, *x, *scaler), (cb_a, csc, acc, va, vs));
        }
        other => panic!("reduce not lowered to LLKReduce, got {other:?}"),
    }
    // No SSA tile-compute ops survive.
    let mut leftover = 0;
    let mut op_id = k.head;
    while !op_id.is_null() {
        if matches!(k.at(op_id), Op::TT(TTOp::MatmulTile { .. } | TTOp::ReduceTile { .. })) {
            leftover += 1;
        }
        op_id = k.next_op(op_id);
    }
    assert_eq!(leftover, 0, "no SSA tile-compute ops survive storage lowering");
}
