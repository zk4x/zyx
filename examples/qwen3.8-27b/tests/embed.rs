// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `embed_kernel(vocab, dim, seq)`: gather embedding
//! out [seq, dim] = w[ids[s], d]

use qwen3_8_27b::embed_kernel;
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

const VOCAB: i64 = 248320;
const DIM: i64 = 5120;
const SEQ: i64 = 6;

#[test]
fn embed() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_embed.safetensors")?;
    let w = goldens["w"].to(dev)?;
    let ids_t = goldens["ids"].to(dev)?;
    let expected = &goldens["output"];
    let kk = embed_kernel(VOCAB, DIM, SEQ);
    let k = kk.compile()?;
    let out = k.forward(&[&w, &ids_t], vec![[SEQ, DIM]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for (&a, &b) in v.iter().zip(exp.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    eprintln!("embed max_err {max_err}");
    assert!(max_err < 1e-3, "embed max_err {max_err}");
    Ok(())
}
