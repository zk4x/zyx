// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `gemm_kernel(M_PAD, HIDDEN, DT_RANK)`: b and a projections
//! out [M_PAD, DT_RANK] = A [M_PAD, HIDDEN] @ B [DT_RANK, HIDDEN].T  (F16, F32 accum).

use qwen3_8_27b::{gemm_kernel, DT_RANK, HIDDEN, M_PAD};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn gemm_ba() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_gemm_ba.safetensors")?;
    let a = goldens["input"].to(dev)?;
    let w = goldens["weight"].to(dev)?;
    let expected = &goldens["output"];
    let kk = gemm_kernel(M_PAD, HIDDEN, DT_RANK);
    let k = kk.compile()?;
    let out = k.forward(&[&a, &w], vec![[M_PAD, DT_RANK]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
        assert!(
            (a - b).abs() < 1e-2,
            "gemm_ba[{i}] = {a}, expected {b}, max_err {max_err}"
        );
    }
    eprintln!("gemm_ba max_err {max_err}");
    Ok(())
}
