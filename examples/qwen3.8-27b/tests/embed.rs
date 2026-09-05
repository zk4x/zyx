// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Embedding gather reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_embed.safetensors` from `tests/embed_ref.py`.
//! Run the dump first: `cd tests && python3.12 embed_ref.py`.

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};
use zyx_nn::Embedding;

#[test]
fn embed() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_embed.safetensors")?;
    let weight = goldens["weight"].to(Dev::Cuda(0))?;
    let input_ids = goldens["input_ids"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let emb = Embedding::from_params(weight)?;
    let out = emb.forward(input_ids)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-5, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}
