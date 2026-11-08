// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama's embedding gather on `Dev::TT(0)`.
//!
//! The exact llama-3.2-3B shape `examples/llama` builds: `weight
//! [128256, 3072]`, `input_ids [1, 1]` (a `seq_len = 1` decode step), output
//! `[1, 1, 3072]`. `Embedding::forward` is that example's chain — `arange ==
//! idx`, cast, broadcast multiply by the weight, sum over the vocab axis — which
//! `fold_loops` lowers to `acc = weight[index]`, the scalar row gather.

use zyx::{DType, Dev, Tensor, ZyxError};
use zyx_nn::Embedding;

#[test]
fn llama_embedding() -> Result<(), ZyxError> {
    // `LlamaConfig::default` in `examples/llama/src/lib.rs`.
    const VOCAB: i64 = 128256;
    const HIDDEN: i64 = 3072;
    // "Hello" under the llama-3 tokenizer (`examples/llama/tests/layer0.rs`).
    const TOKEN: u32 = 9906;

    // Same test on any backend: `ZYX_DEV=cuda:0` is the reference run, and the
    // default is the card.
    let dev: Dev = std::env::var("ZYX_DEV").ok().and_then(|s| s.parse().ok()).unwrap_or(Dev::TT(0));

    let weight = Tensor::rand([VOCAB, HIDDEN], DType::F32)?;
    let expected: Vec<f32> = weight.narrow(0, i64::from(TOKEN), 1i64)?.reshape([HIDDEN])?.try_into()?;

    let mut embedding = Embedding::from_params(weight)?;
    embedding.weight = embedding.weight.to(dev)?;
    let ids = Tensor::from([TOKEN]).reshape([1i64, 1i64])?.to(dev)?;

    let out = embedding.forward(ids)?.to(Dev::C)?;
    let got: Vec<f32> = out.try_into()?;
    assert_eq!(got, expected);
    Ok(())
}

/// The same gather at a size that lets the output be read by hand: a ramp
/// weight, so every datum names its own position.
#[test]
fn embedding_gather_small() -> Result<(), ZyxError> {
    const VOCAB: i64 = 128;
    const HIDDEN: i64 = 32;
    const TOKEN: u32 = 7;

    let dev: Dev = std::env::var("ZYX_DEV").ok().and_then(|s| s.parse().ok()).unwrap_or(Dev::TT(0));
    let ramp: Vec<f32> = (0..VOCAB * HIDDEN).map(|i| i as f32).collect();
    let weight = Tensor::from(ramp).reshape([VOCAB, HIDDEN])?;
    let expected: Vec<f32> = weight.narrow(0, i64::from(TOKEN), 1i64)?.reshape([HIDDEN])?.try_into()?;

    let mut embedding = Embedding::from_params(weight)?;
    embedding.weight = embedding.weight.to(dev)?;
    let mut ids_host = vec![0u32; 16];
    ids_host[0] = TOKEN;
    let ids = Tensor::from(ids_host).reshape([1i64, 16i64])?.to(dev)?;
    let got: Vec<f32> = embedding.forward(ids)?.to(Dev::C)?.try_into()?;
    assert_eq!(got, expected);
    Ok(())
}
