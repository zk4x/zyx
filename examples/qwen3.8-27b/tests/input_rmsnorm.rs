// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `input_rmsnorm_kernel`: out = x * rsqrt(mean(x^2) + eps) * weight
use qwen3_8_27b::{input_rmsnorm_kernel, HIDDEN, S};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn input_rmsnorm() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens =
        Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_input_rmsnorm.safetensors")?;
    let x = goldens["x"].to(dev)?;
    let w = goldens["w"].to(dev)?;
    let expected = &goldens["output"];
    let kk = input_rmsnorm_kernel();
    let k = kk.compile()?;
    let out = k.forward(&[&x, &w], vec![[S, HIDDEN]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (&a, &b) in v.iter().zip(exp.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("input_rmsnorm max_err {max_err}");
    assert!(max_err < 1e-3);
    Ok(())
}
