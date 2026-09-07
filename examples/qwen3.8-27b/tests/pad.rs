// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `pad_kernel(s, m, d)`: pad [s, d] F32 input to [m, d] F16, zero-fill rows s..m.
//! Two uses in qwen3.8-27b: (s=S=6, m=M_PAD=16, d=HIDDEN=5120) for input;
//! (s=S=6, m=M_PAD=16, d=VAL_DIM=6144) for normed.

use qwen3_8_27b::{pad_kernel, HIDDEN, M_PAD, S, VAL_DIM};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn pad_input() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_input.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let expected = &goldens["output"];
    let kk = pad_kernel(S, M_PAD, HIDDEN);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&input], vec![[M_PAD, HIDDEN]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 {
        flops as f64 / total_us / 1e3
    } else {
        0.0
    };
    let gbs = if total_us > 0.0 {
        (read + write) as f64 / total_us / 1e3
    } else {
        0.0
    };
    eprintln!("pad_input forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<zyx::f16> = out[0].to_vec()?;
    let v: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
    let exp: Vec<zyx::f16> = expected.to_vec()?;
    let exp: Vec<f32> = exp.iter().map(|&x| x.to_f32()).collect();
    assert_eq!(v.len(), exp.len());
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "pad_input[{i}] = {a}, expected {b}");
    }
    Ok(())
}

#[test]
fn pad_normed() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_pad_normed.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let expected = &goldens["output"];
    let kk = pad_kernel(S, M_PAD, VAL_DIM);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&input], vec![[M_PAD, VAL_DIM]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 {
        flops as f64 / total_us / 1e3
    } else {
        0.0
    };
    let gbs = if total_us > 0.0 {
        (read + write) as f64 / total_us / 1e3
    } else {
        0.0
    };
    eprintln!("pad_normed forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<zyx::f16> = out[0].to_vec()?;
    let v: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
    let exp: Vec<zyx::f16> = expected.to_vec()?;
    let exp: Vec<f32> = exp.iter().map(|&x| x.to_f32()).collect();
    assert_eq!(v.len(), exp.len());
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "pad_normed[{i}] = {a}, expected {b}");
    }
    Ok(())
}
