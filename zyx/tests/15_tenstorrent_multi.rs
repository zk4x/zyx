// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent multi-op cones: combinations of tile ops in one kernel.
//!
//! Single-op coverage lives in `14_tenstorrent_single.rs`. Everything here
//! exercises the `Programmed` anchor/cursor across kind switches, CB-chained
//! phases, or repeated ops. Silicon rules that constrain these shapes
//! (second-exp-per-episode poison, F16 trig) are documented in
//! `examples/qwen3.8-27b/tenstorrent_debugging.md`.

#![cfg(feature = "tenstorrent")]

use zyx::kernel::{BOp, Dev, Kernel, MemScope, TileReduceKind};
use zyx::{DType, Tensor, ZyxError};

/// Probe: add then add (same BINARY op twice, no exp anywhere).
/// Green (0/1024): the second-exp poison is exp-specific, not
/// "second op" in general.
#[test]
fn tenstorrent_probe_add_add() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    let tb = k.load_global_tile(b, 0);
    k.store_circular(cb, tb, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let vb = k.load_circular(cb, 0);
    let t = k.add(va, vb);
    let s = k.add(t, va);
    k.store_circular(cout, s, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data_a: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let data_b: Vec<f32> = (0..32 * 32).map(|j| ((j + 7) % 32) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let a_t = to_tt(data_a.clone())?;
    let b_t = to_tt(data_b.clone())?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, ((&x, &y), &v)) in data_a.iter().zip(data_b.iter()).zip(z.iter()).enumerate() {
        let expected = (x + y) + x;
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("add-add bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Probe: neg then exp (exp is the 2nd op, but the 1st exp).
/// Green (0/1024): an exp after another op is fine; only the 2nd exp
/// per episode breaks.
#[test]
fn tenstorrent_probe_neg_exp() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let en = k.neg(va);
    let e = k.exp(en);
    k.store_circular(cout, e, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(data.clone(), [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&x, &v)) in data.iter().zip(z.iter()).enumerate() {
        let expected = (-x).exp();
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("neg-exp bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (exp then add): proves the `Programmed` phase
/// cursor — `exp` hoists as the anchor, `add_binary_tile_init` goes
/// inline once at the unary→binary switch. Single exp only: two exps
/// per episode are broken on silicon (see debugging doc), so the fair
/// switch test is `exp(a)+b`.
#[test]
fn tenstorrent_mixed_exp_add() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    let tb = k.load_global_tile(b, 0);
    k.store_circular(cb, tb, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let ea = k.exp(va);
    let vb = k.load_circular(cb, 0);
    let s = k.add(ea, vb);
    k.store_circular(cout, s, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 2): F16-exact steps, no overflow.
    let data_a: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let data_b: Vec<f32> = (0..32 * 32).map(|j| ((j + 7) % 32) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let a_t = to_tt(data_a.clone())?;
    let b_t = to_tt(data_b.clone())?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, ((&x, &y), &v)) in data_a.iter().zip(data_b.iter()).zip(z.iter()).enumerate() {
        let expected = x.exp() + y;
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("mixed bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// sin then sin in F16.
/// IGNORED: F16 trig is broken on silicon (identity passthrough —
/// single sin returns its inputs; see debugging doc). Repetition for
/// sin is untested (needs a BF16 probe); this test documents the F16
/// failure, it does not test our codegen.
#[test]
#[ignore]
fn tenstorrent_probe_sin_sin() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    k.barrier();
    let va = k.load_circular(ca, 0);
    let sa = k.sin(va);
    let sb = k.sin(sa);
    k.store_circular(cout, sb, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(data.clone(), [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&x, &v)) in data.iter().zip(z.iter()).enumerate() {
        let expected = x.sin().sin();
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("sin-sin bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (transpose then exp): Transpose→Unary switch.
#[test]
fn tenstorrent_mixed_transpose_exp() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let tx = k.load_global_tile(x, 0);
    k.store_circular(cin, tx, 0);
    k.barrier();
    let va = k.load_circular(cin, 0);
    let t = k.transpose_tile(va);
    let e = k.exp(t);
    k.store_circular(cout, e, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 2): F16-exact steps, no overflow.
    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 32) as f32 * 0.0625).collect();
    let x_t = Tensor::from_vec(data.clone(), [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&x_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let expected = data[c * 32 + r].exp();
            if (z[r * 32 + c] - expected).abs() >= 3e-2 {
                if bad < 10 {
                    println!("z[{r}][{c}] = {}, expected {expected}", z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("mixed transpose-exp bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (exp then column-sum): Unary→Reduce switch,
/// the softmax-sum pattern. The exp drains through a CB: reduce
/// takes CB tile loads only.
#[test]
fn tenstorrent_mixed_exp_reduce() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let x = k.param(DType::F16);
    let s = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let cin = k.circular_storage(DType::F16, 1);
    let csc = k.circular_storage(DType::F16, 1);
    let cmid = k.circular_storage(DType::F16, 1);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    let tx = k.load_global_tile(x, 0);
    k.store_circular(cin, tx, 0);
    let ts = k.load_global_tile(s, 0);
    k.store_circular(csc, ts, 0);
    k.barrier();
    let acc = k.storage(DType::F16, MemScope::Register, 1024);
    let va = k.load_circular(cin, 0);
    let e = k.exp(va);
    k.store_circular(cmid, e, 0);
    let ve = k.load_circular(cmid, 0);
    let vs = k.load_circular(csc, 0);
    let av = k.load_register_tile(acc, 0);
    let f = k.reduce_tile(ve, vs, av, BOp::Add, TileReduceKind::Col);
    k.store_register_tile(acc, f, 0);
    let g = k.load_register_tile(acc, 0);
    k.store_circular(cout, g, 0);
    k.barrier();
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Inputs in [0, 1): column sums stay small in F16.
    let data: Vec<f32> = (0..32 * 32).map(|j| (j % 16) as f32 * 0.0625).collect();
    let to_tt = |v: Vec<f32>, rows: i64, cols: i64| -> Result<Tensor, ZyxError> {
        Tensor::from_vec(v, [rows, cols])?.tilize()?.cast(DType::F16).to(Dev::TT(0))
    };
    let x_t = to_tt(data.clone(), 32, 32)?;
    let s_t = to_tt(vec![1.0f32; 1024], 32, 32)?;
    let out_bufs = compiled.forward(&[&x_t, &s_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for c in 0..32 {
        let expected: f32 = (0..32).map(|r| data[r * 32 + c].exp()).sum();
        if (z[c] - expected).abs() >= 5e-1 {
            if bad < 10 {
                println!("z[{c}] = {}, expected {expected}", z[c]);
            }
            bad += 1;
        }
    }
    println!("mixed exp-reduce bad: {bad} / 32");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (matmul then bias-add): Matmul→Binary switch.
/// Same geometry as `tenstorrent_matmul_single_core` plus a per-nt
/// bias tile added to each acc cone.
#[test]
fn tenstorrent_mixed_matmul_bias() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let c = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 1);
    let cb = k.circular_storage(DType::F16, 1);
    let cc = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt,
    // C bias tile (mt,nt) at mt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 2, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
            let ct = k.mad(mti, 2, nti);
            let cbase = k.mad(ct, 1024, 0);
            let tc = k.load_global_tile(c, cbase);
            k.store_circular(cc, tc, nti);
        });
    });
    k.barrier();
    // Compute: one acc cone per output tile, bias added after the Kt
    // accumulation steps.
    k.loop_over(1, |k, _mti| {
        k.loop_over(2, |k, nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            let vc = k.load_circular(cc, nti);
            let s = k.add(f, vc);
            k.store_circular(cout, s, 0);
        });
    });
    k.barrier();
    // Writer: output tile (mt,nt) at mt*Nt+nt, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(2, |k, nti| {
            let ot = k.mad(mti, 2, nti);
            let obase = k.mad(ot, 1024, 0);
            let v = k.load_circular(cout, 0);
            k.store_global_tile(out, v, obase);
        });
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // A[32,64] @ B[64,64] + C[32,64], values kept small.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 64).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let c_data: Vec<f32> = (0..32 * 64).map(|j| (j % 8) as f32 * 0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 64];
    for r in 0..32 {
        for c in 0..64 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 64 + c];
            }
            expected[r * 64 + c] = s + c_data[r * 64 + c];
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let c_t = Tensor::from_vec(c_data, [32, 64])?.tilize()?.cast(DType::F32).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t, &c_t], vec![[32, 64]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 64)?.to_vec()?;
    assert_eq!(z.len(), 2048);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("mixed mm-bias bad: {bad} / 2048");
    assert_eq!(bad, 0);

    Ok(())
}
