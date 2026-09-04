// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! LM head reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_lm_head.safetensors` from
//! `tests/lm_head_ref.py`. Run the dump first:
//! `cd tests && python3.12 lm_head_ref.py`.

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};
use zyx_nn::Linear;

#[test]
fn lm_head() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let proj = Linear { weight: goldens["weight"].to(Dev::Cuda(0))?, bias: None };
    let input = goldens["input"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let out = proj.forward(input)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-4, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}
