// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `qk_norm_kernel`: per-head norm on q and k
use qwen3_8_27b::{KD, S, qk_norm_kernel};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn qk_norm() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_qk_norm.safetensors")?;
    let q = goldens["q"].to(dev)?;
    let k = goldens["k"].to(dev)?;
    let qw = goldens["qw"].to(dev)?;
    let kw = goldens["kw"].to(dev)?;
    let expected_q = &goldens["q_out"];
    let expected_k = &goldens["k_out"];
    let kk = qk_norm_kernel(24, 4, KD);
    let (flops, read, write) = kk.flop_mem_rw();
    let ck = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = ck.forward(&[&q, &k, &qw, &kw], vec![[24, S, KD], [4, S, KD]])?;
    out[0].sync()?;
    out[1].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("qk_norm forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let qv: Vec<f32> = out[0].to_vec()?;
    let kv: Vec<f32> = out[1].to_vec()?;
    let eq: Vec<f32> = expected_q.to_vec()?;
    let ek: Vec<f32> = expected_k.to_vec()?;
    assert_eq!(qv.len(), eq.len());
    assert_eq!(kv.len(), ek.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in qv.iter().zip(eq.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    for (i, (&a, &b)) in kv.iter().zip(ek.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("qk_norm max_err {max_err}");
    assert!(max_err < 1e-3);
    Ok(())
}
