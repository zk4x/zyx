// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `mlp_kernel(m, inter)`: out [m, inter] = silu(gate) * up

use qwen3_8_27b::{M_PAD, mlp_kernel};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

const INTERMEDIATE: i64 = 17408;

#[test]
fn mlp() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_mlp.safetensors")?;
    let gate = goldens["gate"].to(dev)?;
    let up = goldens["up"].to(dev)?;
    let expected = &goldens["output"];
    let kk = mlp_kernel(M_PAD, INTERMEDIATE);
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&gate, &up], vec![[M_PAD, INTERMEDIATE]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("mlp forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("mlp max_err {max_err}");
    assert!(max_err < 1e-3, "mlp max_err {max_err}");
    Ok(())
}
