// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `input_rmsnorm_kernel`: out = x * rsqrt(mean(x^2) + eps) * weight
use qwen3_8_27b::{HIDDEN, S, input_rmsnorm_kernel};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn input_rmsnorm() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_input_rmsnorm.safetensors")?;
    let x = goldens["x"].to(dev)?;
    let w = goldens["w"].to(dev)?;
    let expected = &goldens["output"];
    let kk = input_rmsnorm_kernel();
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&x, &w], vec![[S, HIDDEN]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("input_rmsnorm forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("input_rmsnorm max_err {max_err}");
    assert!(max_err < 1e-3);
    Ok(())
}
