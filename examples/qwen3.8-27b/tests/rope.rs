// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `rope_kernel(seq, heads, head_dim, rot_dim)`: partial RoPE
//! out [H*S, D] = rope(x [H*S, D], cos [S, rot_dim], sin [S, rot_dim])
//! Qwen3.5-27B full-attn dims: heads=24, head_dim=256, rot_dim=64.

use qwen3_8_27b::rope_kernel;
use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

const S: i64 = 6;
const HEADS: i64 = 24;
const HEAD_DIM: i64 = 256;
const ROT_DIM: i64 = 64;

#[test]
fn rope() -> Result<(), ZyxError> {
    let dev = Dev::Cuda(0);
    let goldens = Tensor::load("/home/x/Dev/rust/zyx/examples/data/qwen3_rope.safetensors")?;
    let x = goldens["x"].to(dev)?;
    let cos = goldens["cos"].to(dev)?;
    let sin = goldens["sin"].to(dev)?;
    let expected = &goldens["output"];
    let kk = rope_kernel(S, HEADS, HEAD_DIM, ROT_DIM);
    let k = kk.compile()?;
    let out = k.forward(&[&x, &cos, &sin], vec![[HEADS * S, HEAD_DIM]])?;
    let v: Vec<f32> = out[0].to_vec()?;
    let exp: Vec<f32> = expected.to_vec()?;
    assert_eq!(v.len(), exp.len());
    let mut max_err = 0f32;
    for i in 0..v.len() {
        max_err = max_err.max((v[i] - exp[i]).abs());
    }
    eprintln!("rope max_err {max_err}");
    assert!(max_err < 1e-3, "rope max_err {max_err}");
    Ok(())
}
