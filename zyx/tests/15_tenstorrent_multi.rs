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
