// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `gemm_kernel(M_PAD, HIDDEN, CONV_DIM)`: qkv projection
//! out [M_PAD, CONV_DIM] = A [M_PAD, HIDDEN] @ B [CONV_DIM, HIDDEN].T  (F16, F32 accum).

use qwen3_8_27b::{CONV_DIM, HIDDEN, M_PAD, gemm_kernel, gemm_tt};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn gemm_qkv() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_gemm_qkv.safetensors")?;
    let a = goldens["input"].to(dev)?;
    let w = goldens["weight"].to(dev)?;
    let expected = &goldens["output"];
    let kk = gemm_kernel(M_PAD, HIDDEN, CONV_DIM);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&a, &w], vec![[M_PAD, CONV_DIM]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("gemm_qkv forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
        assert!((a - b).abs() < 1e-2, "gemm_qkv[{i}] = {a}, expected {b}, max_err {max_err}");
    }
    eprintln!("gemm_qkv max_err {max_err}");
    Ok(())
}

/// `gemm_tt(32, HIDDEN, CONV_DIM)` on the P100A: F16 tilized A [32, HIDDEN]
/// (16 golden rows + 16 zero rows) @ F16 KN-tilized W [HIDDEN, CONV_DIM]
/// -> F32 tilized out [32, CONV_DIM], DST-accumulated over Kt in one
/// kernel (official `matmul_single_core` structure).
/// First 16 rows vs the torch golden (1e-2, like CUDA); zero-padded rows
/// must read back ~0 (catches accumulation leaking across output tiles).
#[test]
fn gemm_qkv_tt() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_gemm_qkv.safetensors")?;
    let a_f16: Vec<zyx::f16> = goldens["input"].to_vec()?;
    let w_f16: Vec<zyx::f16> = goldens["weight"].to_vec()?;
    let expected = &goldens["output"];
    const M32: i64 = 32;
    const NT: i64 = CONV_DIM / 32;
    // A: 16 golden rows + 16 zero rows.
    let mut a_pad = vec![zyx::f16::from_f32(0.0); (M32 * HIDDEN) as usize];
    a_pad[..(M_PAD * HIDDEN) as usize].copy_from_slice(&a_f16);
    // W [N, K] row-major -> [K, N] (KN tile order) host-side.
    let (n, k) = (CONV_DIM as usize, HIDDEN as usize);
    let mut wt = vec![zyx::f16::from_f32(0.0); k * n];
    for ni in 0..n {
        for ki in 0..k {
            wt[ki * n + ni] = w_f16[ni * k + ki];
        }
    }
    let a_t = Tensor::from_vec(a_pad, [M32, HIDDEN])?.tilize()?.to(dev)?;
    let w_t = Tensor::from_vec(wt, [HIDDEN, CONV_DIM])?.tilize()?.to(dev)?;
    let z_t = Tensor::from_vec(vec![0.0f32; 1024], [32, 32])?.tilize()?.to(dev)?;

    let kk = gemm_tt(M32, HIDDEN, CONV_DIM);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    // REVIEW-THEN-LAUNCH (AGENTS.md): with ZYX_TT_DUMP_ONLY=1, stop after
    // compile so generated sources (ZYX_DEBUG=16) can be compared against
    // the official tt-metal kernels before anything executes on the board.
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&a_t, &w_t, &z_t], vec![[NT * 1024]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("gemm_qkv_tt forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let flat: Vec<f32> = out[0].to_vec()?;
    let back = Tensor::from_vec(flat, [M32, CONV_DIM])?.untilize(M32, CONV_DIM)?;
    let v: Vec<f32> = back.to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), 2 * exp.len());
    let mut max_err = 0f32;
    for i in 0..exp.len() {
        max_err = max_err.max((v[i] - exp[i]).abs());
        assert!((v[i] - exp[i]).abs() < 1e-2, "gemm_qkv_tt[{i}] = {}, expected {}, max_err {max_err}", v[i], exp[i]);
    }
    let mut max_zero = 0f32;
    for i in exp.len()..v.len() {
        max_zero = max_zero.max(v[i].abs());
    }
    eprintln!("gemm_qkv_tt max_err {max_err}, zero-rows max {max_zero}");
    assert!(max_zero < 1e-2, "gemm_qkv_tt zero-rows max {max_zero}");
    Ok(())
}

/// `gemm_tt(64, HIDDEN, CONV_DIM)` on the P100A: Mt=2 tile rows over the
/// same goldens (16 golden rows + 48 zero rows). First 16 output rows vs
/// the torch golden (1e-2); zero-padded rows must read back ~0 (catches
/// accumulation leaking across output tiles, now also across tile rows).
/// Exercises the pack-scope acquire: per output tile, not outermost loop.
#[test]
fn gemm_qkv_tt_mt2() -> Result<(), ZyxError> {
    let dev = Dev::TT(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_gemm_qkv.safetensors")?;
    let a_f16: Vec<zyx::f16> = goldens["input"].to_vec()?;
    let w_f16: Vec<zyx::f16> = goldens["weight"].to_vec()?;
    let expected = &goldens["output"];
    const M64: i64 = 64;
    const MT: i64 = M64 / 32;
    const NT: i64 = CONV_DIM / 32;
    // A: 16 golden rows + 48 zero rows.
    let mut a_pad = vec![zyx::f16::from_f32(0.0); (M64 * HIDDEN) as usize];
    a_pad[..(M_PAD * HIDDEN) as usize].copy_from_slice(&a_f16);
    // W [N, K] row-major -> [K, N] (KN tile order) host-side.
    let (n, k) = (CONV_DIM as usize, HIDDEN as usize);
    let mut wt = vec![zyx::f16::from_f32(0.0); k * n];
    for ni in 0..n {
        for ki in 0..k {
            wt[ki * n + ni] = w_f16[ni * k + ki];
        }
    }
    let a_t = Tensor::from_vec(a_pad, [M64, HIDDEN])?.tilize()?.to(dev)?;
    let w_t = Tensor::from_vec(wt, [HIDDEN, CONV_DIM])?.tilize()?.to(dev)?;
    let z_t = Tensor::from_vec(vec![0.0f32; 1024], [32, 32])?.tilize()?.to(dev)?;

    let kk = gemm_tt(M64, HIDDEN, CONV_DIM);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    // REVIEW-THEN-LAUNCH (AGENTS.md): with ZYX_TT_DUMP_ONLY=1, stop after
    // compile so generated sources (ZYX_DEBUG=16) can be compared against
    // the official tt-metal kernels before anything executes on the board.
    if std::env::var("ZYX_TT_DUMP_ONLY").is_ok() {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&a_t, &w_t, &z_t], vec![[(MT * NT * 1024)]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("gemm_qkv_tt_mt2 forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let flat: Vec<f32> = out[0].to_vec()?;
    let back = Tensor::from_vec(flat, [M64, CONV_DIM])?.untilize(M64, CONV_DIM)?;
    let v: Vec<f32> = back.to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), 4 * exp.len());
    let mut max_err = 0f32;
    for i in 0..exp.len() {
        max_err = max_err.max((v[i] - exp[i]).abs());
        assert!((v[i] - exp[i]).abs() < 1e-2, "gemm_qkv_tt_mt2[{i}] = {}, expected {}, max_err {max_err}", v[i], exp[i]);
    }
    let mut max_zero = 0f32;
    for i in exp.len()..v.len() {
        max_zero = max_zero.max(v[i].abs());
    }
    eprintln!("gemm_qkv_tt_mt2 max_err {max_err}, zero-rows max {max_zero}");
    assert!(max_zero < 1e-2, "gemm_qkv_tt_mt2 zero-rows max {max_zero}");
    Ok(())
}
