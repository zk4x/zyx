// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! RMSNorm at the llama prefill shape: F16 `[1, 2, 3072]` on the graph path,
//! checked against an inline host reference.

use std::result::Result;
use zyx::{DType, Tape, Tensor, ZyxError};

#[test]
fn rmsnorm_large_rank3() -> Result<(), ZyxError> {
    const H: usize = 3072;
    let x = Tensor::randn([1i64, 2, H as i64], DType::F16)?;
    let scale = Tensor::from(vec![1.0f32; H]).cast(DType::F16);
    let eps = Tensor::from(1e-5f32).cast(DType::F16);

    let tape = Tape::empty();
    tape.add(&scale)?;
    // Same op mix as RMSNorm::forward.
    let xx = x.clone() * x.clone();
    let mean = xx.mean_keepdim([-1])?;
    let normed = x.clone() * (mean + eps).rsqrt() * scale;
    tape.realize([&normed])?;

    let xv: Vec<f32> = x.clone().cast(DType::F32).try_into()?;
    let got: Vec<f32> = normed.clone().cast(DType::F32).try_into()?;
    assert_eq!(xv.len(), 2 * H);
    assert_eq!(got.len(), 2 * H);
    let mut maxd = 0f32;
    for r in 0..2 {
        let mut ms = 0f64;
        for i in 0..H {
            let v = xv[r * H + i] as f64;
            ms += v * v;
        }
        ms = ms / H as f64 + 1e-5;
        let inv = 1.0 / ms.sqrt();
        for i in 0..H {
            let want = xv[r * H + i] as f64 * inv;
            maxd = maxd.max((got[r * H + i] as f64 - want).abs() as f32);
        }
    }
    eprintln!("rmsnorm_large_rank3 maxdiff={maxd}");
    assert!(maxd < 5e-2, "RMSNorm diverged at [1,2,3072] F16: {maxd}");
    Ok(())
}

#[test]
fn rmsnorm_large_rank3_eager() -> Result<(), ZyxError> {
    const H: usize = 3072;
    let x = Tensor::randn([1i64, 2, H as i64], DType::F16)?;
    let scale = Tensor::from(vec![1.0f32; H]).cast(DType::F16);
    let eps = Tensor::from(1e-5f32).cast(DType::F16);

    // Same op mix, no tape: eager custom-kernel path.
    let xx = x.clone() * x.clone();
    let mean = xx.mean_keepdim([-1])?;
    let normed = x.clone() * (mean + eps).rsqrt() * scale;

    let xv: Vec<f32> = x.clone().cast(DType::F32).try_into()?;
    let got: Vec<f32> = normed.clone().cast(DType::F32).try_into()?;
    assert_eq!(xv.len(), 2 * H);
    assert_eq!(got.len(), 2 * H);
    let mut maxd = 0f32;
    for r in 0..2 {
        let mut ms = 0f64;
        for i in 0..H {
            let v = xv[r * H + i] as f64;
            ms += v * v;
        }
        ms = ms / H as f64 + 1e-5;
        let inv = 1.0 / ms.sqrt();
        for i in 0..H {
            let want = xv[r * H + i] as f64 * inv;
            maxd = maxd.max((got[r * H + i] as f64 - want).abs() as f32);
        }
    }
    eprintln!("rmsnorm_large_rank3_eager maxdiff={maxd}");
    assert!(maxd < 5e-2, "Eager RMSNorm diverged at [1,2,3072] F16: {maxd}");
    Ok(())
}
