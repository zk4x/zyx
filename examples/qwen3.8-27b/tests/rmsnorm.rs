// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! RMSNorm reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_rmsnorm.safetensors` from
//! `tests/rmsnorm_ref.py`. Run the dump first:
//! `cd tests && python3.12 rmsnorm_ref.py`.

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};
use zyx_nn::RMSNorm;

#[test]
fn rmsnorm() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_rmsnorm.safetensors")?;
    let scale = goldens["scale"].to(Dev::Cuda(0))?;
    let input = goldens["input"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    // Qwen3_5 dumps scale = 1 + weight (zero-centered convention).
    let norm = RMSNorm { scale, eps: 1e-6 };
    let out = norm.forward(input)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-5, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}
