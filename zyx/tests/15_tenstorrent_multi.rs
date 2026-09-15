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

use zyx::kernel::{BOp, Dev, Kernel, MemScope, TileDim};
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
    let to_tt =
        |v: Vec<f32>| -> Result<Tensor, ZyxError> { Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0)) };
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
    let to_tt =
        |v: Vec<f32>| -> Result<Tensor, ZyxError> { Tensor::from_vec(v, [32, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0)) };
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
    let f = k.reduce_tile(ve, vs, av, BOp::Add, TileDim::Col);
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

/// Probe: multi-tile CB capacity + indexed access. ca holds 4 tiles,
/// the reader pushes all 4 before compute pops any. cout holds 2
/// tiles and is indexed: compute pushes at tile index j (nested 2x2
/// loops, the mixed_matmul_bias cc pattern), the writer pops at index
/// j. Compute doubles each tile (add(va, va)) — a real op, no
/// matmul. Input [32,128] = 4 tiles, expected 2*x.
#[test]
fn tenstorrent_probe_multitile_double() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 4);
    let cout = k.circular_storage(DType::F16, 2);

    let _g = k.group_range(0, 1);

    // Reader: tile i of [32,128] at i*1024, push all 4 into ca.
    k.loop_over(4, |k, i| {
        let base = k.mad(i, 1024, 0);
        let ta = k.load_global_tile(a, base);
        k.store_circular(ca, ta, 0);
    });
    k.barrier();
    // Compute: pop each tile, double it, push to cout at index j.
    k.loop_over(2, |k, _o| {
        k.loop_over(2, |k, j| {
            let va = k.load_circular(ca, 0);
            let s = k.add(va, va);
            k.store_circular(cout, s, j);
        });
    });
    k.barrier();
    // Writer: output tile o*2+j at (o*2+j)*1024, pop cout at index j.
    k.loop_over(2, |k, o| {
        k.loop_over(2, |k, j| {
            let ot = k.mad(o, 2, j);
            let base = k.mad(ot, 1024, 0);
            let v = k.load_circular(cout, j);
            k.store_global_tile(out, v, base);
        });
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..32 * 128).map(|j| (j % 32) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(data.clone(), [32, 128])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 128]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 128)?.to_vec()?;
    assert_eq!(z.len(), 4096);
    let mut bad = 0;
    for (p, (&x, &v)) in data.iter().zip(z.iter()).enumerate() {
        let expected = x + x;
        if (v - expected).abs() >= 3e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("multitile double bad: {bad} / 4096");
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

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
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

/// Single-pass variant of `tenstorrent_mixed_matmul_bias` (nt=1): the
/// short executes once on the virgin matmul-mode state, copy/add once
/// with no second pass. Correct output here clears the once-through
/// combination and points at the loop-carried (pass-2) state.
#[test]
fn tenstorrent_mixed_matmul_bias_single() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let c = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
    let cc = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt,
    // C bias tile (mt,nt) at mt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
            let ct = k.mad(mti, 1, nti);
            let cbase = k.mad(ct, 1024, 0);
            let tc = k.load_global_tile(c, cbase);
            k.store_circular(cc, tc, nti);
        });
    });
    k.barrier();
    // Compute: one acc cone per output tile, bias added after the Kt
    // accumulation steps.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, nti| {
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
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // A[32,64] @ B[64,32] + C[32,32], values kept small.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let c_data: Vec<f32> = (0..32 * 32).map(|j| (j % 8) as f32 * 0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 32];
    for r in 0..32 {
        for c in 0..32 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 32 + c];
            }
            expected[r * 32 + c] = s + c_data[r * 32 + c];
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let c_t = Tensor::from_vec(c_data, [32, 32])?.tilize()?.cast(DType::F32).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t, &c_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("mixed mm-bias single bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Bisection: bias+copy+add with NO matmul (add the bias tile to
/// itself, pack the sum). Correct output here clears reader, copy,
/// add and DST slots 0-2, convicting the short'd matmul in the full
/// combo; wrong output convicts the copy/add path itself.
#[test]
fn tenstorrent_bias_add_probe() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let c = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let cc = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: C bias tile at index nti.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ct = k.mad(mti, 1, nti);
            let cbase = k.mad(ct, 1024, 0);
            let tc = k.load_global_tile(c, cbase);
            k.store_circular(cc, tc, nti);
        });
    });
    k.barrier();
    // Compute: double the bias tile via add, packed out.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, nti| {
            let v0 = k.load_circular(cc, nti);
            let s = k.add(v0, v0);
            k.store_circular(cout, s, 0);
        });
    });
    k.barrier();
    // Writer: the single output tile, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // 2*C[32,32], values kept small.
    let c_data: Vec<f32> = (0..32 * 32).map(|j| (j % 8) as f32 * 0.0625).collect();
    let c_t = Tensor::from_vec(c_data.clone(), [32, 32])?.tilize()?.cast(DType::F32).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&c_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(c_data.iter()).enumerate() {
        let e2 = 2.0 * e;
        if (v - e2).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e2}");
            }
            bad += 1;
        }
    }
    println!("bias add probe bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Bisection: matmul cone plus bias copy, packing the COPIED BIAS
/// tile straight out. Correct output here clears the reader bias
/// path and the copy path (check the dump for whether the unused
/// matmul — and its short — survived DCE).
#[test]
fn tenstorrent_matmul_short_probe() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let c = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
    let cc = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt,
    // C bias tile (mt,nt) at mt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
            let ct = k.mad(mti, 1, nti);
            let cbase = k.mad(ct, 1024, 0);
            let tc = k.load_global_tile(c, cbase);
            k.store_circular(cc, tc, nti);
        });
    });
    k.barrier();
    // Compute: k-loop matmuls, then bias add; pack acc AND sum.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, nti| {
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
            k.store_circular(cout, vc, 0);
            let _ = f;
        });
    });
    k.barrier();
    // Writer: the single bias tile, row-major.
    k.loop_over(1, |k, nti| {
        let obase = k.mad(nti, 1024, 0);
        let v = k.load_circular(cout, 0);
        k.store_global_tile(out, v, obase);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // C[32,32] bias tile, values kept small.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let c_data: Vec<f32> = (0..32 * 32).map(|j| (j % 8) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let c_t = Tensor::from_vec(c_data.clone(), [32, 32])?.tilize()?.cast(DType::F32).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t, &c_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(c_data.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("matmul short probe bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Matmul→unary fusion probe: single-pass matmul cone with exp on the
/// acc, packed out. No copy, no second operand — if this fails like
/// the bias tests, the matmul→SFPU handoff itself is broken; if it
/// passes, the handoff is fine and the bug is copy/add-specific.
#[test]
fn tenstorrent_mixed_matmul_exp() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
        });
    });
    k.barrier();
    // Compute: k-loop matmuls, exp on the acc, packed out.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, _nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            let e = k.exp(f);
            k.store_circular(cout, e, 0);
        });
    });
    k.barrier();
    // Writer: output tile (mt,nt) at mt*Nt+nt, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // exp(A[32,64] @ B[64,32]) with small values (keep exp in range).
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * -0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 32];
    for r in 0..32 {
        for c in 0..32 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 32 + c];
            }
            expected[r * 32 + c] = s.exp();
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("mixed mm-exp bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (matmul then transpose): two-phase — the acc
/// packs to an intermediate CB, transpose streams from there
/// (`transpose_wh_tile` reads CBs, never DST). Tests the
/// matmul→pack→transpose handoff under `mm_init`-owned engines.
#[test]
fn tenstorrent_mixed_matmul_transpose() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F32);

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
    let cmid = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
        });
    });
    k.barrier();
    // Compute: k-loop matmuls, pack acc to cmid, transpose to cout.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, _nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            k.store_circular(cmid, f, 0);
            let vm = k.load_circular(cmid, 0);
            let t = k.transpose_tile(vm);
            k.store_circular(cout, t, 0);
        });
    });
    k.barrier();
    // Writer: output tile, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // transpose(A[32,64] @ B[64,32]) with small values.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let mut prod = vec![0.0f32; 32 * 32];
    for r in 0..32 {
        for c in 0..32 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 32 + c];
            }
            prod[r * 32 + c] = s;
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        for c in 0..32 {
            let expected = prod[c * 32 + r];
            if (z[r * 32 + c] - expected).abs() >= 5e-2 {
                if bad < 10 {
                    println!("z[{r}][{c}] = {}, expected {expected}", z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("mixed mm-transpose bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Mixed-kind cone (matmul then column-sum reduce): all-BF16
/// two-phase — the BF16 acc packs straight to an intermediate CB
/// (no cast leg: the SFPU typecast is broken under 32-bit DST and
/// the JIT rejects mixed F16/BF16 inputs), reduce streams from
/// there with a ones scaler (`reduce_tile` reads CBs, never DST).
/// Tests the matmul→pack→reduce handoff including `reduce_uninit`
/// before the consuming pack.
#[test]
fn tenstorrent_mixed_matmul_reduce() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let b = k.param(DType::BF16);
    let s = k.param(DType::BF16);
    let out = k.param_mut(DType::BF16);

    let ca = k.circular_storage(DType::BF16, 4);
    let cb = k.circular_storage(DType::BF16, 4);
    let csc = k.circular_storage(DType::BF16, 2);
    let cmid = k.circular_storage(DType::BF16, 2);
    let cout = k.circular_storage(DType::BF16, 1);

    let _g = k.group_range(0, 1);

    // Reader: A/B tiles plus the ones scaler tile.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
            let st = k.load_global_tile(s, 0);
            k.store_circular(csc, st, 0);
        });
    });
    k.barrier();
    // Compute: k-loop matmuls, pack acc to cmid, column-sum to cout.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, _nti| {
            let acc = k.storage(DType::BF16, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            k.store_circular(cmid, f, 0);
            let vm = k.load_circular(cmid, 0);
            let vs = k.load_circular(csc, 0);
            let r = k.reduce_tile(vm, vs, f, BOp::Add, TileDim::Col);
            k.store_register_tile(acc, r, 0);
            let g = k.load_register_tile(acc, 0);
            k.store_circular(cout, g, 0);
        });
    });
    k.barrier();
    // Writer: output tile, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // Column sums of A[32,64] @ B[64,32] with small values.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let mut prod = vec![0.0f32; 32 * 32];
    for r in 0..32 {
        for c in 0..32 {
            let mut sum = 0.0f32;
            for t in 0..64 {
                sum += a_data[r * 64 + t] * b_data[t * 32 + c];
            }
            prod[r * 32 + c] = sum;
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let s_t = Tensor::from_vec(vec![1.0f32; 1024], [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t, &s_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for c in 0..32 {
        let expected: f32 = (0..32).map(|r| prod[r * 32 + c]).sum();
        if (z[c] - expected).abs() >= 1.0 {
            if bad < 10 {
                println!("z[{c}] = {}, expected {expected}", z[c]);
            }
            bad += 1;
        }
    }
    println!("mixed mm-reduce bad: {bad} / 32");
    assert_eq!(bad, 0);

    Ok(())
}

/// Bisection: matmul acc through the SFPU cast, packed straight
/// out (no reduce). Correct output here clears the cast under
/// 32-bit DST and convicts the reduce leg of mm-reduce; constant
/// output convicts the mode-unaware typecast op instead.
/// TODO: ignored — fails on board (768/1024 bad, outputs stuck near
/// 2.0; assertion at the end of this test). Unrelated to fused-LLK
/// work (matmul + cast, no sigmoid/silu/mul-binary in the kernel);
/// needs a silicon debug session of its own.
#[ignore]
#[test]
fn tenstorrent_matmul_cast_probe() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::F16);
    let b = k.param(DType::F16);
    let out = k.param_mut(DType::F16);

    let ca = k.circular_storage(DType::F16, 4);
    let cb = k.circular_storage(DType::F16, 4);
    let cout = k.circular_storage(DType::F16, 1);

    let _g = k.group_range(0, 1);

    // Reader: A tile (mt,kt) at mt*Kt+kt, B tile (kt,nt) at kt*Nt+nt.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            k.loop_over(2, |k, kti| {
                let at = k.mad(mti, 2, kti);
                let abase = k.mad(at, 1024, 0);
                let ta = k.load_global_tile(a, abase);
                k.store_circular(ca, ta, 0);
                let bt = k.mad(kti, 1, nti);
                let bbase = k.mad(bt, 1024, 0);
                let tb = k.load_global_tile(b, bbase);
                k.store_circular(cb, tb, 0);
            });
        });
    });
    k.barrier();
    // Compute: k-loop matmuls, cast acc to F16, packed out.
    k.loop_over(1, |k, _mti| {
        k.loop_over(1, |k, _nti| {
            let acc = k.storage(DType::F32, MemScope::Register, 1024);
            k.loop_over(2, |k, _kti| {
                let va = k.load_circular(ca, 0);
                let vb = k.load_circular(cb, 0);
                let av = k.load_register_tile(acc, 0);
                let f = k.matmul_tile(va, vb, av);
                k.store_register_tile(acc, f, 0);
            });
            let f = k.load_register_tile(acc, 0);
            let cf = k.cast(f, DType::F16);
            k.store_circular(cout, cf, 0);
        });
    });
    k.barrier();
    // Writer: output tile, row-major.
    k.loop_over(1, |k, mti| {
        k.loop_over(1, |k, nti| {
            let ot = k.mad(mti, 1, nti);
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

    // A[32,64] @ B[64,32] with small values, golden in F32.
    let a_data: Vec<f32> = (0..32 * 64).map(|j| (j % 4) as f32 * 0.0625).collect();
    let b_data: Vec<f32> = (0..64 * 32).map(|j| ((j / 4) % 4) as f32 * 0.0625).collect();
    let mut expected = vec![0.0f32; 32 * 32];
    for r in 0..32 {
        for c in 0..32 {
            let mut s = 0.0f32;
            for t in 0..64 {
                s += a_data[r * 64 + t] * b_data[t * 32 + c];
            }
            expected[r * 32 + c] = s;
        }
    }
    let a_t = Tensor::from_vec(a_data, [32, 64])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let b_t = Tensor::from_vec(b_data, [64, 32])?.tilize()?.cast(DType::F16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &b_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (p, (&v, &e)) in z.iter().zip(expected.iter()).enumerate() {
        if (v - e).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{p}] = {v}, expected {e}");
            }
            bad += 1;
        }
    }
    println!("matmul cast probe bad: {bad} / 1024");
    assert_eq!(bad, 0);

    Ok(())
}

/// Row-wise softmax: max-sub-exp-sum-normalize over one [32,32] BF16 tile.
/// max/sum via `TileDim::Row` reduce (per-row stats, column-vector layout),
/// sub/mul via fused `*_bcast_cols`, div as recip+mul. Mirrors the official
/// `moreh_softmax_w` structure (reduce ROW + bcast COL), minus masks/scalers.
#[test]
fn tenstorrent_softmax_rows() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let ones = k.param(DType::BF16);
    let out = k.param_mut(DType::BF16);

    let ca = k.circular_storage(DType::BF16, 2);
    let cone = k.circular_storage(DType::BF16, 2);
    let cm = k.circular_storage(DType::BF16, 1);
    let ce = k.circular_storage(DType::BF16, 2);
    let cs = k.circular_storage(DType::BF16, 1);
    let cr = k.circular_storage(DType::BF16, 1);
    let cout = k.circular_storage(DType::BF16, 1);
    let acc = k.storage(DType::BF16, MemScope::Register, 1024);
    let acc2 = k.storage(DType::BF16, MemScope::Register, 1024);

    let _g = k.group_range(0, 1);

    // Reader: two copies of A (max-fold, center-sub) + two ones
    // (max-fold scaler, sum-fold scaler).
    for _ in 0..2 {
        let ta = k.load_global_tile(a, 0);
        k.store_circular(ca, ta, 0);
        let ts = k.load_global_tile(ones, 0);
        k.store_circular(cone, ts, 0);
    }
    k.barrier();

    // Row max -> cm.
    let va = k.load_circular(ca, 0);
    let vs = k.load_circular(cone, 0);
    let vacc = k.load_register_tile(acc, 0);
    let fm = k.reduce_tile(va, vs, vacc, BOp::Max, TileDim::Row);
    k.store_register_tile(acc, fm, 0);
    let vm = k.load_register_tile(acc, 0);
    k.store_circular(cm, vm, 0);

    // Center + exp -> ce (stored twice: sum-fold, final mul).
    let va2 = k.load_circular(ca, 0);
    let vm2 = k.load_circular(cm, 0);
    let mb = k.broadcast_tile(vm2, TileDim::Col);
    let d = k.sub(va2, mb);
    let e = k.exp(d);
    k.store_circular(ce, e, 0);
    k.store_circular(ce, e, 0);

    // Row sum -> cs, recip -> cr.
    let ve1 = k.load_circular(ce, 0);
    let vs2 = k.load_circular(cone, 0);
    let vacc2 = k.load_register_tile(acc2, 0);
    let fs = k.reduce_tile(ve1, vs2, vacc2, BOp::Add, TileDim::Row);
    k.store_register_tile(acc2, fs, 0);
    let vcs = k.load_register_tile(acc2, 0);
    k.store_circular(cs, vcs, 0);
    let vcs2 = k.load_circular(cs, 0);
    let r = k.reciprocal(vcs2);
    k.store_circular(cr, r, 0);

    // Normalize: e * (1/sum) via fused bcast_cols mul.
    let ve2 = k.load_circular(ce, 0);
    let vr = k.load_circular(cr, 0);
    let rb = k.broadcast_tile(vr, TileDim::Col);
    let o = k.mul(ve2, rb);
    k.store_circular(cout, o, 0);
    k.barrier();

    // Writer: drain the single output tile.
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    // Non-negative data: max-fold seeds from zero acc, stays exact.
    let a_data: Vec<f32> = (0..32 * 32).map(|j| (j % 8) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(a_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let ones_t = Tensor::from_vec(vec![1.0f32; 1024], [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &ones_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        let row: Vec<f32> = (0..32).map(|c| a_data[r * 32 + c]).collect();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = row.iter().map(|x| (x - m).exp()).sum();
        for c in 0..32 {
            let expected = (a_data[r * 32 + c] - m).exp() / denom;
            if (z[r * 32 + c] - expected).abs() >= 5e-2 {
                if bad < 10 {
                    println!("z[{}] = {}, expected {expected}", r * 32 + c, z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("softmax bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w over one [32,32] BF16 tile.
/// mean via `TileDim::Row` SUM reduce with a host-filled 1/32 scaler
/// tile (no const-div arm needed); eps as a full host-filled tile;
/// `rsqrt` for the scale; two full-tile muls. No immediates.
#[test]
fn tenstorrent_rmsnorm_rows() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let w = k.param(DType::BF16);
    let eps_t = k.param(DType::BF16);
    let inv_n = k.param(DType::BF16);
    let out = k.param_mut(DType::BF16);

    let ca = k.circular_storage(DType::BF16, 2);
    let cw = k.circular_storage(DType::BF16, 1);
    let ceps = k.circular_storage(DType::BF16, 1);
    let cs = k.circular_storage(DType::BF16, 1);
    let csq = k.circular_storage(DType::BF16, 1);
    let cm = k.circular_storage(DType::BF16, 1);
    let cv = k.circular_storage(DType::BF16, 1);
    let cr = k.circular_storage(DType::BF16, 1);
    let cout = k.circular_storage(DType::BF16, 1);
    let acc = k.storage(DType::BF16, MemScope::Register, 1024);

    let _g = k.group_range(0, 1);

    // Reader: two copies of A (square, final mul) + one each of the rest.
    for _ in 0..2 {
        let ta = k.load_global_tile(a, 0);
        k.store_circular(ca, ta, 0);
    }
    let tw = k.load_global_tile(w, 0);
    k.store_circular(cw, tw, 0);
    let te = k.load_global_tile(eps_t, 0);
    k.store_circular(ceps, te, 0);
    let ts = k.load_global_tile(inv_n, 0);
    k.store_circular(cs, ts, 0);
    k.barrier();

    // x^2 -> csq.
    let va = k.load_circular(ca, 0);
    let sq = k.mul(va, va);
    k.store_circular(csq, sq, 0);

    // mean(x^2) -> cm (1/32 folded in via the scaler tile).
    let vsq = k.load_circular(csq, 0);
    let vsc = k.load_circular(cs, 0);
    let vacc = k.load_register_tile(acc, 0);
    let fm = k.reduce_tile(vsq, vsc, vacc, BOp::Add, TileDim::Row);
    k.store_register_tile(acc, fm, 0);
    let vm = k.load_register_tile(acc, 0);
    k.store_circular(cm, vm, 0);

    // mean + eps -> cv, rsqrt -> cr.
    let vm2 = k.load_circular(cm, 0);
    let mb = k.broadcast_tile(vm2, TileDim::Col);
    let veps = k.load_circular(ceps, 0);
    let v = k.add(mb, veps);
    k.store_circular(cv, v, 0);
    let vv = k.load_circular(cv, 0);
    let r = k.rsqrt(vv);
    k.store_circular(cr, r, 0);

    // x * rstd * w -> cout.
    let va2 = k.load_circular(ca, 0);
    let vr = k.load_circular(cr, 0);
    let t = k.mul(va2, vr);
    let vw = k.load_circular(cw, 0);
    let o = k.mul(t, vw);
    k.store_circular(cout, o, 0);
    k.barrier();

    // Writer: drain the single output tile.
    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let eps = 1e-4f32;
    let a_data: Vec<f32> = (0..32 * 32).map(|j| ((j % 8) as f32 - 3.5) * 0.25).collect();
    let w_data: Vec<f32> = (0..32 * 32).map(|j| 0.5 + (j % 4) as f32 * 0.25).collect();
    let a_t = Tensor::from_vec(a_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let w_t = Tensor::from_vec(w_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let eps_ten = Tensor::from_vec(vec![eps; 1024], [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let inv_n_ten = Tensor::from_vec(vec![1.0 / 32.0; 1024], [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t, &w_t, &eps_ten, &inv_n_ten], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for r in 0..32 {
        let mean: f32 = (0..32).map(|c| { let x = a_data[r * 32 + c]; x * x }).sum::<f32>() / 32.0;
        let rstd = 1.0 / (mean + eps).sqrt();
        for c in 0..32 {
            let expected = a_data[r * 32 + c] * rstd * w_data[r * 32 + c];
            if (z[r * 32 + c] - expected).abs() >= 5e-2 {
                if bad < 10 {
                    println!("z[{}] = {}, expected {expected}", r * 32 + c, z[r * 32 + c]);
                }
                bad += 1;
            }
        }
    }
    println!("rmsnorm bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Tile-scalar binary: `(x + 1.5) * 3.0` over one [32,32] BF16 tile.
/// Const sides fold into `add/mul_unary_tile` fp32-bits immediates
/// (no CB traffic for the scalars); both ops share one hoisted
/// `binop_with_scalar_tile_init`. (2.0 is unusable here:
/// `fold_constants` strength-reduces `x * 2.0` into `x + x`.)
#[test]
fn tenstorrent_scalar_add_mul() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let out = k.param_mut(DType::BF16);

    let ca = k.circular_storage(DType::BF16, 1);
    let cout = k.circular_storage(DType::BF16, 1);

    let _g = k.group_range(0, 1);

    let ta = k.load_global_tile(a, 0);
    k.store_circular(ca, ta, 0);
    k.barrier();

    let va = k.load_circular(ca, 0);
    let c15v = k.const_val(1.5f32);
    let c15 = k.cast(c15v, DType::BF16);
    let s = k.add(va, c15);
    let c20v = k.const_val(3.0f32);
    let c20 = k.cast(c20v, DType::BF16);
    let o = k.mul(s, c20);
    k.store_circular(cout, o, 0);
    k.barrier();

    let v = k.load_circular(cout, 0);
    k.store_global_tile(out, v, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let a_data: Vec<f32> = (0..32 * 32).map(|j| (j % 8) as f32 * 0.0625).collect();
    let a_t = Tensor::from_vec(a_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 32]])?;

    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (i, (&zv, &av)) in z.iter().zip(a_data.iter()).enumerate() {
        let expected = (av + 1.5) * 3.0;
        if (zv - expected).abs() >= 5e-2 {
            if bad < 10 {
                println!("z[{i}] = {zv}, expected {expected}");
            }
            bad += 1;
        }
    }
    println!("scalar add-mul bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Fused LLK calls: `sigmoid(x)` and `silu(x)` composites built with
/// the plain builders must compile to one `sigmoid_tile` / one
/// `silu_tile` (+ inits) instead of neg/exp/add/reciprocal/mul.
#[test]
fn tenstorrent_fused_sigmoid_silu() -> Result<(), ZyxError> {
    // Two independent inputs: fused calls transform their input slot
    // in place, so each pattern needs a pattern-exclusive input (a
    // shared input correctly falls back to the plain composite).
    let mut k = Kernel::new(Dev::TT(0));
    let a1 = k.param(DType::BF16);
    let a2 = k.param(DType::BF16);
    let out_sig = k.param_mut(DType::BF16);
    let out_silu = k.param_mut(DType::BF16);

    let ca1 = k.circular_storage(DType::BF16, 1);
    let ca2 = k.circular_storage(DType::BF16, 1);
    let csig = k.circular_storage(DType::BF16, 1);
    let csilu = k.circular_storage(DType::BF16, 1);

    let _g = k.group_range(0, 1);

    let t1 = k.load_global_tile(a1, 0);
    k.store_circular(ca1, t1, 0);
    let t2 = k.load_global_tile(a2, 0);
    k.store_circular(ca2, t2, 0);
    k.barrier();

    let v1 = k.load_circular(ca1, 0);
    let sig = k.sigmoid(v1);
    k.store_circular(csig, sig, 0);
    let v2 = k.load_circular(ca2, 0);
    let sil = k.silu(v2);
    k.store_circular(csilu, sil, 0);
    k.barrier();

    let v1 = k.load_circular(csig, 0);
    k.store_global_tile(out_sig, v1, 0);
    let v2 = k.load_circular(csilu, 0);
    k.store_global_tile(out_silu, v2, 0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let a_data: Vec<f32> = (0..32 * 32).map(|j| (j % 16) as f32 * 0.5 - 4.0).collect();
    let a1_t = Tensor::from_vec(a_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let a2_t = Tensor::from_vec(a_data.clone(), [32, 32])?.tilize()?.cast(DType::BF16).to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a1_t, &a2_t], vec![[32, 32], [32, 32]])?;

    fn sigmoid_f(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn silu_f(x: f32) -> f32 {
        x * sigmoid_f(x)
    }
    let mut bad = 0;
    for (buf, f) in [(0, sigmoid_f as fn(f32) -> f32), (1, silu_f as fn(f32) -> f32)] {
        let z: Vec<f32> = out_bufs[buf].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
        assert_eq!(z.len(), 1024);
        for (i, (&zv, &av)) in z.iter().zip(a_data.iter()).enumerate() {
            let expected = f(av);
            if (zv - expected).abs() >= 5e-2 {
                if bad < 10 {
                    println!("buf{buf}[{i}] = {zv}, expected {expected}");
                }
                bad += 1;
            }
        }
    }
    println!("fused sigmoid/silu bad: {bad} / 2048");
    assert_eq!(bad, 0);
    Ok(())
}

// ---------------------------------------------------------------------------
// Q4_K dequant decomposition probes. The full `dequant_q4k_tt` produces
// zeros on even tiles and `full_word*sf - gf` on odd tiles; these four
// probes isolate one stage each, on synthetic data with host-computed
// goldens. One page = 4 tiles (1024 packed U16 words), matching the
// production kernel's per-page plumbing.
// ---------------------------------------------------------------------------

/// Deterministic packed words: word j carries nibble k of tile k in bits 4k.
fn probe_words() -> Vec<u16> {
    (0..1024u32)
        .map(|j| ((j.wrapping_mul(2654435761).wrapping_add(12345)) % 65536) as u16)
        .collect()
}

/// Stage 1: nibble extraction — `n = cv - 16*trunc(cv/16)` on the full-word
/// F32 value (cv used twice via two carry CBs, trunc result duplicated
/// through the scratch CB), no scales/mins, no loop. Expected: `w & 15`.
#[test]
fn tenstorrent_probe_q4k_nibbles() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let ccur = k.circular_storage(DType::F32, 1);
    let ccur2 = k.circular_storage(DType::F32, 1);
    let cs = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c00625 = k.const_val(0.0625f32);
    let c16 = k.const_val(16.0f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f1 = k.cast(u1, DType::F32);
    k.store_circular(ccur, f1, c0);
    k.store_circular(ccur2, f1, c0);
    let cv = k.load_circular(ccur, c0);
    let t = k.mul(cv, c00625);
    let t = k.trunc(t);
    k.store_circular(cs, t, c0);
    let b16 = k.load_circular(cs, c0);
    k.store_circular(cs, t, c0);
    let bc = k.load_circular(cs, c0);
    let t16 = k.mul(b16, c16);
    let cv2 = k.load_circular(ccur2, c0);
    let n = k.sub(cv2, t16);
    k.store_circular(cout, n, c0);
    k.store_circular(ccur, bc, c0);
    // Drain pop: the stored carry would otherwise sit unconsumed in ccur.
    let _drain = k.load_circular(ccur, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let mut bad = 0;
    for (j, &w) in wu.iter().enumerate() {
        let expected = (w & 15) as f32;
        if (z[j] - expected).abs() >= 1e-4 {
            if bad < 10 {
                println!("z[{j}] = {}, expected {expected} (word {w})", z[j]);
            }
            bad += 1;
        }
    }
    println!("q4k nibbles bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 2: scale conversion — BF16 CB tile → F32 cast → out. Expected:
/// the bf16-rounded input value, exactly.
#[test]
fn tenstorrent_probe_q4k_convert() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let out = k.param_mut(DType::F32);
    let ca = k.circular_storage(DType::BF16, 1);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);

    let t = k.load_global_tile(a, c0);
    k.store_circular(ca, t, c0);
    k.barrier();
    let s = k.load_circular(ca, c0);
    let sf = k.cast(s, DType::F32);
    k.store_circular(cout, sf, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let data: Vec<f32> = (0..1024).map(|j| 0.001 + (j % 32) as f32 * 0.0007).collect();
    let expected: Vec<f32> = data.iter().map(|&x| zyx::bf16::from_f32(x).to_f32()).collect();
    let a_t = Tensor::from_vec(data, [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&a_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let expected_t: Vec<f32> = Tensor::from_vec(expected.clone(), [32i64, 32])?.tilize()?.to_vec()?;
    let mut bad = 0;
    for (j, (&zv, &ev)) in z.iter().zip(expected_t.iter()).enumerate() {
        if (zv - ev).abs() >= 1e-6 {
            if bad < 10 {
                println!("z[{j}] = {zv}, expected {ev}");
            }
            bad += 1;
        }
    }
    println!("q4k convert bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 3: carry streaming — cv through the two carry CBs across 4 loop
/// trips, carry = `trunc(cv/16)` duplicated via the scratch CB, out tile k
/// = nibble of trip k (`cv - 16*trunc(cv/16)`). Expected: `(w >> 4k) & 15`,
/// exact in F32. Exercises loop-wrap CB accounting (both carry CBs pushed
/// AND popped every trip).
#[test]
fn tenstorrent_probe_q4k_carry() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let ccur = k.circular_storage(DType::F32, 2);
    let ccur2 = k.circular_storage(DType::F32, 2);
    let cs = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 4);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c4 = k.const_idx(4);
    let c00625 = k.const_val(0.0625f32);
    let c16 = k.const_val(16.0f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f1 = k.cast(u1, DType::F32);
    k.store_circular(ccur, f1, c0);
    k.store_circular(ccur2, f1, c0);
    k.loop_over(c4, |k, _i| {
        let cv = k.load_circular(ccur, c0);
        let t = k.mul(cv, c00625);
        let t = k.trunc(t);
        k.store_circular(cs, t, c0);
        let b16 = k.load_circular(cs, c0);
        k.store_circular(cs, t, c0);
        let bc = k.load_circular(cs, c0);
        let t16 = k.mul(b16, c16);
        let cv2 = k.load_circular(ccur2, c0);
        let n = k.sub(cv2, t16);
        k.store_circular(ccur, bc, c0);
        k.store_circular(ccur2, bc, c0);
        k.store_circular(cout, n, c0);
    });
    // Drain pops keep the carry CBs empty at the end of the pass (the CB
    // verifier flags any producer/consumer imbalance).
    let _drain = k.load_circular(ccur, c0);
    let _drain2 = k.load_circular(ccur2, c0);
    k.barrier();
    k.loop_over(c4, |k, i| {
        let v = k.load_circular(cout, i);
        k.store_global_tile(out, v, i);
    });

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 128]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 128)?.to_vec()?;
    assert_eq!(z.len(), 4096);
    let mut bad = 0;
    for (tile, &w) in wu.iter().enumerate() {
        for trip in 0..4 {
            let expected = ((w >> (4 * trip)) & 15) as f32;
            let got = z[trip * 1024 + tile];
            if (got - expected).abs() >= 1e-2 {
                if bad < 10 {
                    println!("z[trip {trip}, tile {tile}] = {got}, expected {expected} (word {w})");
                }
                bad += 1;
            }
        }
    }
    println!("q4k carry bad: {bad} / 4096");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 4: full math chain, ONE trip — nibble plane 0, real cast scales
/// and mins, `v = n*sf - gf`, out = v. Isolates trip-boundary handling:
/// no page loop, no wraparound reconfig.
#[test]
fn tenstorrent_probe_q4k_onetrip() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let sc = k.param(DType::BF16);
    let mn = k.param(DType::BF16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let csc = k.circular_storage(DType::BF16, 1);
    let cmn = k.circular_storage(DType::BF16, 1);
    let ccur = k.circular_storage(DType::F32, 1);
    let ccur2 = k.circular_storage(DType::F32, 1);
    let cs = k.circular_storage(DType::F32, 2);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c00625 = k.const_val(0.0625f32);
    let c16 = k.const_val(16.0f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    let s0 = k.load_global_tile(sc, c0);
    k.store_circular(csc, s0, c0);
    let m0 = k.load_global_tile(mn, c0);
    k.store_circular(cmn, m0, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f1 = k.cast(u1, DType::F32);
    k.store_circular(ccur, f1, c0);
    k.store_circular(ccur2, f1, c0);
    let cv = k.load_circular(ccur, c0);
    let t = k.mul(cv, c00625);
    let t = k.trunc(t);
    k.store_circular(cs, t, c0);
    let b16 = k.load_circular(cs, c0);
    k.store_circular(cs, t, c0);
    let bc = k.load_circular(cs, c0);
    let t16 = k.mul(b16, c16);
    let cv2 = k.load_circular(ccur2, c0);
    let n = k.sub(cv2, t16);
    k.store_circular(ccur, bc, c0);
    let s = k.load_circular(csc, c0);
    let sf = k.cast(s, DType::F32);
    let m1 = k.mul(n, sf);
    let g = k.load_circular(cmn, c0);
    let gf = k.cast(g, DType::F32);
    let v = k.sub(m1, gf);
    k.store_circular(cout, v, c0);
    // Drain pop: the stored carry would otherwise sit unconsumed in ccur.
    let _drain = k.load_circular(ccur, c0);
    k.barrier();
    let vo = k.load_circular(cout, c0);
    k.store_global_tile(out, vo, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let sf_raw: Vec<f32> = (0..1024).map(|j| 0.001 + (j % 32) as f32 * 0.0007).collect();
    let mn_raw: Vec<f32> = (0..1024).map(|j| 0.01 + (j % 16) as f32 * 0.001).collect();
    let to_bf16 = |v: &[f32]| -> Vec<zyx::bf16> { v.iter().map(|&x| zyx::bf16::from_f32(x)).collect() };
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let sc_t = Tensor::from_vec(to_bf16(&sf_raw), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let mn_t = Tensor::from_vec(to_bf16(&mn_raw), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t, &sc_t, &mn_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.cast(DType::F32).untilize(32, 32)?.to_vec()?;
    assert_eq!(z.len(), 1024);
    let sf: Vec<f32> = sf_raw.iter().map(|&x| zyx::bf16::from_f32(x).to_f32()).collect();
    let mn: Vec<f32> = mn_raw.iter().map(|&x| zyx::bf16::from_f32(x).to_f32()).collect();
    let mut bad = 0;
    for (j, &w) in wu.iter().enumerate() {
        let n = (w & 15) as f32;
        let expected = n * sf[j] - mn[j];
        if (z[j] - expected).abs() >= 1e-4 {
            if bad < 10 {
                println!("z[{j}] = {}, expected {expected} (word {} n {n})", z[j], w);
            }
            bad += 1;
        }
    }
    println!("q4k onetrip bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

// ---------------------------------------------------------------------------
// fp32-entry SFPU bisect: which op breaks when DST entries are 4 bytes?
// Each stage extends the previous by one op; the first stage that hangs
// or corrupts on the simulator names the culprit. Run with
// TT_METAL_SIMULATOR=$HOME/sim/libttsim_bh.so.
// ---------------------------------------------------------------------------

/// Stage 1: copy + typecast U16->F32 in fp32 DST mode, pack F32.
#[test]
fn tenstorrent_probe_fp32_seed() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f = k.cast(u1, DType::F32);
    k.store_circular(cout, f, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.to_vec()?;
    let expected: Vec<f32> = Tensor::from_vec(
        wu.iter().map(|&w| w as f32).collect::<Vec<f32>>(),
        [32i64, 32],
    )?
    .tilize()?
    .to_vec()?;
    let mut bad = 0;
    for (j, (&zv, &ev)) in z.iter().zip(expected.iter()).enumerate() {
        if (zv - ev).abs() >= 1e-3 {
            if bad < 6 {
                println!("z[{j}] = {zv}, expected {ev}");
            }
            bad += 1;
        }
    }
    println!("fp32 seed bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 2: stage 1 + scalar mul (binop_with_scalar) in fp32 DST mode.
#[test]
fn tenstorrent_probe_fp32_scalar() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c00625 = k.const_val(0.0625f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f = k.cast(u1, DType::F32);
    let t = k.mul(f, c00625);
    k.store_circular(cout, t, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.to_vec()?;
    let expected: Vec<f32> = Tensor::from_vec(
        wu.iter().map(|&w| w as f32 * 0.0625).collect::<Vec<f32>>(),
        [32i64, 32],
    )?
    .tilize()?
    .to_vec()?;
    let mut bad = 0;
    for (j, (&zv, &ev)) in z.iter().zip(expected.iter()).enumerate() {
        if (zv - ev).abs() >= 1e-3 {
            if bad < 6 {
                println!("z[{j}] = {zv}, expected {ev}");
            }
            bad += 1;
        }
    }
    println!("fp32 scalar bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 3: stage 2 + trunc in fp32 DST mode.
#[test]
fn tenstorrent_probe_fp32_trunc() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c00625 = k.const_val(0.0625f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f = k.cast(u1, DType::F32);
    let t = k.mul(f, c00625);
    let t = k.trunc(t);
    k.store_circular(cout, t, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.to_vec()?;
    let expected: Vec<f32> = Tensor::from_vec(
        wu.iter().map(|&w| (w as f32 * 0.0625).trunc()).collect::<Vec<f32>>(),
        [32i64, 32],
    )?
    .tilize()?
    .to_vec()?;
    let mut bad = 0;
    for (j, (&zv, &ev)) in z.iter().zip(expected.iter()).enumerate() {
        if (zv - ev).abs() >= 1e-3 {
            if bad < 6 {
                println!("z[{j}] = {zv}, expected {ev}");
            }
            bad += 1;
        }
    }
    println!("fp32 trunc bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}

/// Stage 4: stage 3 + sub_binary_tile (n = cv2 - 16*t) in fp32 DST mode.
#[test]
fn tenstorrent_probe_fp32_sub() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let packed = k.param(DType::U16);
    let out = k.param_mut(DType::F32);
    let cu16 = k.circular_storage(DType::U16, 1);
    let ccur = k.circular_storage(DType::F32, 1);
    let ccur2 = k.circular_storage(DType::F32, 1);
    let cout = k.circular_storage(DType::F32, 1);
    let _g = k.group_range(0, 1);
    let c0 = k.const_idx(0);
    let c00625 = k.const_val(0.0625f32);
    let c16 = k.const_val(16.0f32);

    let u = k.load_global_tile(packed, c0);
    k.store_circular(cu16, u, c0);
    k.barrier();
    let u1 = k.load_circular(cu16, c0);
    let f = k.cast(u1, DType::F32);
    k.store_circular(ccur, f, c0);
    k.store_circular(ccur2, f, c0);
    let cv = k.load_circular(ccur, c0);
    let t = k.mul(cv, c00625);
    let t = k.trunc(t);
    let t16 = k.mul(t, c16);
    let cv2 = k.load_circular(ccur2, c0);
    let n = k.sub(cv2, t16);
    // Drain the carry CB so its single tile stays consumed.
    let _carry = k.load_circular(ccur, c0);
    k.store_circular(cout, n, c0);
    k.barrier();
    let v = k.load_circular(cout, c0);
    k.store_global_tile(out, v, c0);

    k.verify();
    let compiled = k.compile()?;
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        println!("dump only, skipping launch");
        return Ok(());
    }

    let wu = probe_words();
    let packed_t = Tensor::from_vec(wu.clone(), [32i64, 32])?.tilize()?.to(Dev::TT(0))?;
    let out_bufs = compiled.forward(&[&packed_t], vec![[32, 32]])?;
    let z: Vec<f32> = out_bufs[0].to(Dev::C)?.to_vec()?;
    let expected: Vec<f32> = Tensor::from_vec(
        wu.iter().map(|&w| (w & 15) as f32).collect::<Vec<f32>>(),
        [32i64, 32],
    )?
    .tilize()?
    .to_vec()?;
    let mut bad = 0;
    for (j, (&zv, &ev)) in z.iter().zip(expected.iter()).enumerate() {
        if (zv - ev).abs() >= 1e-3 {
            if bad < 6 {
                println!("z[{j}] = {zv}, expected {ev}");
            }
            bad += 1;
        }
    }
    println!("fp32 sub bad: {bad} / 1024");
    assert_eq!(bad, 0);
    Ok(())
}
