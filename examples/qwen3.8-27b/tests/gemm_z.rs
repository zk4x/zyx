// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `gemm_kernel(M_PAD, HIDDEN, VAL_DIM)`: z projection
//! out [M_PAD, VAL_DIM] = A [M_PAD, HIDDEN] @ B [VAL_DIM, HIDDEN].T  (F16, F32 accum).

use qwen3_8_27b::{gemm_kernel, HIDDEN, M_PAD, VAL_DIM};
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

#[test]
fn gemm_z() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_gemm_z.safetensors")?;
    let a = goldens["input"].to(dev)?;
    let w = goldens["weight"].to(dev)?;
    let expected = &goldens["output"];
    let kk = gemm_kernel(M_PAD, HIDDEN, VAL_DIM);
    let k = kk.compile()?;
    let out = k.forward(&[&a, &w], vec![[M_PAD, VAL_DIM]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (i, (&a, &b)) in v.iter().zip(exp.iter()).enumerate() {
        max_err = max_err.max((a - b).abs());
        assert!(
            (a - b).abs() < 1e-2,
            "gemm_z[{i}] = {a}, expected {b}, max_err {max_err}"
        );
    }
    eprintln!("gemm_z max_err {max_err}");
    Ok(())
}
