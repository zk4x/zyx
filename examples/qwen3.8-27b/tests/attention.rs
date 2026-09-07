// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `attention_kernel(seq, h, kv, d)`: causal GQA attention with gate
//! out [seq, h*d] = sigmoid(gate) * softmax(QK^T/sqrt(d) + causal_mask) V
//! Qwen3.5-27B: h=24, kv=4, d=256.

use qwen3_8_27b::attention_kernel;
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

const S: i64 = 6;
const H: i64 = 24;
const KV: i64 = 4;
const D: i64 = 256;

#[test]
fn attention() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_attention.safetensors")?;
    let q = goldens["q"].to(dev)?;
    let k = goldens["k"].to(dev)?;
    let v = goldens["v"].to(dev)?;
    let gate = goldens["gate"].to(dev)?;
    let expected = &goldens["output"];
    let kk = attention_kernel(S, H, KV, D);
    let (flops, read, write) = kk.flop_mem_rw();
    let ck = kk.compile()?;
    let t0 = std::time::Instant::now();
    let out = ck.forward(&[&q, &k, &v, &gate], vec![[S, H * D]])?;
    out[0].sync()?;
    let total_us = t0.elapsed().as_micros() as f64;
    let tflops = if total_us > 0.0 { flops as f64 / total_us / 1e3 } else { 0.0 };
    let gbs = if total_us > 0.0 { (read + write) as f64 / total_us / 1e3 } else { 0.0 };
    eprintln!("attention forward+sync {total_us:.0}us, {tflops:.2} TFLOPS, {gbs:.1} GB/s");
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("attention max_err {max_err}");
    assert!(max_err < 1e-3, "attention max_err {max_err}");
    Ok(())
}
