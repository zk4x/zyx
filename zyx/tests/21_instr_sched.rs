// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `instruction_schedule` ordering guarantees, exercised as an external
//! pass author would: build IR with the public kernel API, run the pass,
//! inspect op order through `head`/`at`/`next_op`. No device needed.

use zyx::DType;
use zyx::kernel::{Dev, Kernel, MemScope, Op, OpId, ParamKind};

fn params_storages_in_order(k: &Kernel) -> Vec<(MemScope, bool)> {
    let mut order = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        match k.at(op_id) {
            Op::Param { kind, .. } => match kind {
                ParamKind::Global | ParamKind::Variable => order.push((MemScope::Global, true)),
                ParamKind::GlobalMut => order.push((MemScope::Global, false)),
            },
            Op::Storage { scope, .. } => order.push((*scope, false)),
            _ => {}
        }
        op_id = k.next_op(op_id);
    }
    order
}

fn op_ids_in_order(k: &Kernel) -> Vec<OpId> {
    let mut order = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        order.push(op_id);
        op_id = k.next_op(op_id);
    }
    order
}

#[test]
fn test_instruction_schedule_orders_params_and_storages() {
    let mut k = Kernel::new(Dev::Auto);
    let _local_rw = k.storage(DType::F32, MemScope::Local, 4);
    let global_ro = k.param(DType::F32);
    let _local_ro = k.storage(DType::F32, MemScope::Local, 4);
    let global_rw = k.param_mut(DType::F32);

    let gidx_len = k.const_idx(4);
    let gidx = k.group_range(0, gidx_len);
    let c = k.const_val(1.0f32);
    let load = k.load(global_ro, gidx);
    let add = k.add(load, c);
    k.store(global_rw, add, gidx);

    k.instruction_schedule();

    assert_eq!(
        params_storages_in_order(&k),
        vec![
            (MemScope::Global, true),
            (MemScope::Global, false),
            (MemScope::Local, false),
            (MemScope::Local, false),
        ]
    );

    let order = op_ids_in_order(&k);
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert!(pos(gidx) < pos(load));
    assert!(pos(c) < pos(add));
    assert!(pos(load) < pos(add));
}

#[test]
fn test_instruction_schedule_keeps_stores_in_loops() {
    let mut k = Kernel::new(Dev::Auto);
    let src = k.param(DType::F32);
    let dst = k.param_mut(DType::F32);

    let len = k.const_idx(4u32);
    let mut loop_id = OpId::NULL;
    k.loop_over(len, |k, lv| {
        loop_id = lv;
        let in_loop_load = k.load(src, loop_id);
        let add = k.add(in_loop_load, in_loop_load);
        k.store(dst, add, loop_id);
    });

    k.instruction_schedule();

    let order = op_ids_in_order(&k);
    let store = order.iter().copied().find(|&id| matches!(k.at(id), Op::Store { .. })).unwrap();
    let end_loop = order.iter().copied().find(|&id| matches!(k.at(id), Op::EndLoop)).unwrap();
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert!(pos(loop_id) < pos(store), "store must stay inside its loop");
    assert!(pos(store) < pos(end_loop), "store must stay inside its loop");
}

#[test]
fn test_instruction_schedule_keeps_memory_order_per_target() {
    let mut k = Kernel::new(Dev::Auto);
    let buf = k.param(DType::F32);

    let gidx_len = k.const_idx(4);
    let gidx = k.group_range(0, gidx_len);
    let val = k.const_val(1.0f32);
    k.store(buf, val, gidx);
    let load = k.load(buf, gidx);
    k.store(buf, load, gidx);

    k.instruction_schedule();

    let order = op_ids_in_order(&k);
    let stores: Vec<OpId> = order.iter().copied().filter(|&id| matches!(k.at(id), Op::Store { .. })).collect();
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert_eq!(stores.len(), 2);
    assert!(pos(stores[0]) < pos(load), "load to a target must stay after prior store to it");
    assert!(pos(load) < pos(stores[1]), "load to a target must stay before later store to it");
}

#[test]
fn test_instruction_schedule_keeps_stores_after_barriers() {
    let mut k = Kernel::new(Dev::Auto);
    let buf = k.storage(DType::F32, MemScope::Local, 4);

    let gidx_len = k.const_idx(4);
    let gidx = k.group_range(0, gidx_len);
    let val = k.const_val(1.0f32);
    k.barrier();
    k.store(buf, val, gidx);

    k.instruction_schedule();

    let order = op_ids_in_order(&k);
    let store = order.iter().copied().find(|&id| matches!(k.at(id), Op::Store { .. })).unwrap();
    let barrier = order.iter().copied().find(|&id| matches!(k.at(id), Op::Barrier)).unwrap();
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert!(pos(barrier) < pos(store), "store must stay after the barrier");
}

#[test]
fn test_instruction_schedule_topological() {
    let mut k = Kernel::new(Dev::Auto);
    let src = k.param(DType::F32);
    let dst = k.param_mut(DType::F32);

    let gidx_len = k.const_idx(4);
    let gidx = k.group_range(0, gidx_len);
    let a = k.load(src, gidx);
    let b = k.load(src, gidx);
    let add = k.add(a, b);
    k.store(dst, add, gidx);

    k.instruction_schedule();

    let order = op_ids_in_order(&k);
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert!(pos(a) < pos(add));
    assert!(pos(b) < pos(add));
}

#[test]
fn test_instruction_schedule_never_sinks_across_loops() {
    let mut k = Kernel::new(Dev::Auto);
    let src = k.param(DType::F32);
    let dst = k.param_mut(DType::F32);
    let local = k.storage(DType::F32, MemScope::Local, 4);

    let c0 = k.const_idx(0u32);
    let c5 = k.const_idx(5u32);
    let c4 = k.const_idx(4u32);
    let invariant = k.bit_shift_left(c0, c5);

    let mut loop1 = OpId::NULL;
    let mut loop2 = OpId::NULL;
    k.loop_over(c4, |k, lv| {
        loop1 = lv;
        let idx1 = k.add(invariant, loop1);
        let v1 = k.load(src, idx1);
        k.store(local, v1, idx1);
    });

    k.barrier();

    k.loop_over(c4, |k, lv| {
        loop2 = lv;
        let idx2 = k.add(invariant, loop2);
        let v2 = k.load(local, idx2);
        k.store(dst, v2, idx2);
    });

    k.instruction_schedule();

    let order = op_ids_in_order(&k);
    let pos = |target: OpId| order.iter().position(|&id| id == target).unwrap();
    assert!(pos(invariant) < pos(loop1), "invariant must not be sunk into the first loop");
    assert!(pos(invariant) < pos(loop2), "invariant must not be sunk into the second loop");
}

#[test]
fn _bench_instruction_schedule_large_kernel() {
    let mut k = Kernel::new(Dev::Auto);
    let a = k.param(DType::F32);
    let b = k.param(DType::F32);
    let out = k.param_mut(DType::F32);
    let gidx_len = k.const_idx(1024);
    let gidx = k.group_range(0, gidx_len);
    let mut acc = k.load(a, gidx);
    for _ in 0..200 {
        let x = k.load(b, gidx);
        acc = k.add(acc, x);
        let two = k.const_val(2.0f32);
        let y = k.mul(acc, two);
        acc = k.add(acc, y);
    }
    k.store(out, acc, gidx);

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        k.instruction_schedule();
    }
    let elapsed = start.elapsed();
    println!("1000x schedule on ~800-op kernel: {:?}", elapsed);
}
