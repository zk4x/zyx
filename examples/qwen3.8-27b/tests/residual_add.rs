// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `residual_add_kernel`: out [M_PAD, HIDDEN] = a + b
use qwen3_8_27b::{residual_add_kernel, HIDDEN, M_PAD};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn residual_add() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens =
        Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_residual_add.safetensors")?;
    let a = goldens["a"].to(dev)?;
    let b = goldens["b"].to(dev)?;
    let expected = &goldens["output"];
    let kk = residual_add_kernel();
    let k = kk.compile()?;
    let out = k.forward(&[&a, &b], vec![[M_PAD, HIDDEN]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (&a, &b) in v.iter().zip(exp.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("residual_add max_err {max_err}");
    assert!(max_err < 1e-3);
    Ok(())
}
