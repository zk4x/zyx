// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `conv_silu_kernel`: depthwise causal conv1d (kernel 4, left pad 3) + SiLU
//! out [M_PAD, CONV_DIM] = silu( sum_k convw[c, k] * inp[max(0, t+k-3), c] )

use qwen3_8_27b::{CONV_DIM, M_PAD, conv_silu_kernel};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn conv_silu() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_conv_silu.safetensors")?;
    let input = goldens["input"].to(dev)?;
    let convw = goldens["convw"].to(dev)?;
    let expected = &goldens["output"];
    let kk = conv_silu_kernel();
    let (flops, read, write) = kk.flop_mem_rw();
    let k = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = k.forward(&[&input, &convw], vec![[M_PAD, CONV_DIM]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("conv_silu forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
        assert!((a - b).abs() < 1e-3, "conv_silu[{i}] = {a}, expected {b}, max_err {max_err}");
    }
    eprintln!("conv_silu max_err {max_err}");
    Ok(())
}
