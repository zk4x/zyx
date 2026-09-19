// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `qk_norm_kernel`: per-head norm on q and k
use qwen3_8_27b::{qk_norm_kernel, KD, S};
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
    let ck = kk.compile()?;
    let out = ck.forward(&[&q, &k, &qw, &kw], vec![[24, S, KD], [4, S, KD]])?;
    let qv: Vec<f32> = out[0].to_vec()?;
    let kv: Vec<f32> = out[1].to_vec()?;
    let eq: Vec<f32> = expected_q.to_vec()?;
    let ek: Vec<f32> = expected_k.to_vec()?;
    assert_eq!(qv.len(), eq.len());
    assert_eq!(kv.len(), ek.len());
    let mut max_err = 0f32;
    for (&a, &b) in qv.iter().zip(eq.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    for (&a, &b) in kv.iter().zip(ek.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("qk_norm max_err {max_err}");
    assert!(max_err < 1e-3);
    Ok(())
}
