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

/// The same gather large enough that **both** grid axes exceed the device
/// maximum (the card's `10x11`, the simulator's `10x14`): `12` rows and
/// `512 / 32 = 16` column tiles. `tt_tile` must then split each group into an
/// outer grid the device can dispatch and a `Loop` over the rest, so the
/// launch exercises the grid-loop path rather than a single grid point.
#[test]
fn embedding_gather_small() -> Result<(), ZyxError> {
    const VOCAB: i64 = 128;
    const HIDDEN: i64 = 512;
    const ROWS: i64 = 12;

    let dev: Dev = std::env::var("ZYX_DEV").ok().and_then(|s| s.parse().ok()).unwrap_or(Dev::TT(0));
    let weight = Tensor::rand([VOCAB, HIDDEN], DType::F32)?;
    let tokens: Vec<u32> = (0..ROWS).map(|r| ((r * 37 + 7) % VOCAB) as u32).collect();
    let mut expected: Vec<f32> = Vec::new();
    for &t in tokens.iter() {
        let row: Vec<f32> = weight.narrow(0, i64::from(t), 1i64)?.reshape([HIDDEN])?.try_into()?;
        expected.extend(row);
    }

    let mut embedding = Embedding::from_params(weight.clone())?;
    embedding.weight = embedding.weight.to(dev)?;
    let ids = Tensor::from(tokens.clone()).reshape([ROWS, 1i64])?.to(dev)?;
    let got: Vec<f32> = embedding.forward(ids)?.to(Dev::C)?.try_into()?;
    if got != expected {
        let mut hits: Vec<i64> = Vec::new();
        for r in 0..VOCAB {
            let row: Vec<f32> = weight.narrow(0, r, 1i64)?.reshape([HIDDEN])?.try_into()?;
            if row == got[..(HIDDEN as usize)] {
                hits.push(r);
            }
        }
        panic!("the gather returned row(s) {hits:?} first, expected row {}", tokens[0]);
    }
    Ok(())
}
