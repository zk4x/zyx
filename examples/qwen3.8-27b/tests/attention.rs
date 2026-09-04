// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Full GQA attention reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_attention.safetensors` from
//! `tests/attention_ref.py`. Run the dump first:
//! `cd tests && python3.12 attention_ref.py`.

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};
use zyx_nn::{Linear, RMSNorm};

const H: i64 = 4; // q heads
const KV: i64 = 2; // kv heads
const D: i64 = 8; // head dim
const SEQ: i64 = 4;

fn repeat_kv(xs: &Tensor, n_rep: i64) -> Result<Tensor, ZyxError> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let b: i64 = xs.shape()[0].item();
    let s: i64 = xs.shape()[2].item();
    let d: i64 = xs.shape()[3].item();
    xs.unsqueeze(2)?
        .expand([b, KV, n_rep, s, d])?
        .reshape([b, KV * n_rep, s, d])
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rot_dim: i64) -> Result<Tensor, ZyxError> {
    let last = x.rank() as i32 - 1;
    let head_dim: i64 = x.shape()[last as usize].item();
    let q_rot = x.narrow(last, 0i64, rot_dim)?;
    let q_pass = x.narrow(last, rot_dim, head_dim - rot_dim)?;
    let half = rot_dim / 2;
    let a = q_rot.narrow(last, 0i64, half)?;
    let b = q_rot.narrow(last, half, half)?;
    let neg_b = -&b;
    let rotated = Tensor::cat([&neg_b, &a], last)?;
    let out_rot = &q_rot * cos + &rotated * sin;
    Tensor::cat([&out_rot, &q_pass], last)
}

#[test]
fn attention() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_attention.safetensors")?;
    let dev = Dev::Cuda(0);
    let q_proj = Linear { weight: goldens["q_proj"].to(dev)?, bias: None };
    let k_proj = Linear { weight: goldens["k_proj"].to(dev)?, bias: None };
    let v_proj = Linear { weight: goldens["v_proj"].to(dev)?, bias: None };
    let o_proj = Linear { weight: goldens["o_proj"].to(dev)?, bias: None };
    let q_norm = RMSNorm { scale: goldens["q_scale"].to(dev)?, eps: 1e-6 };
    let k_norm = RMSNorm { scale: goldens["k_scale"].to(dev)?, eps: 1e-6 };
    let cos = goldens["cos"].to(dev)?;
    let sin = goldens["sin"].to(dev)?;
    let input = goldens["input"].to(dev)?;
    let expected = goldens["output"].to_vec::<f32>()?;

    // q_proj output is interleaved per head: [q0, g0, q1, g1, ...].
    // Reshape to heads first, then chunk query/gate.
    let qg = q_proj.forward(&input)?.reshape([1i64, SEQ, H, 2i64 * D])?;
    let q = qg.narrow(-1, 0i64, D)?.reshape([1i64, SEQ, H, D])?;
    let gate = qg.narrow(-1, D, D)?.reshape([1i64, SEQ, H * D])?;
    let q = q_norm.forward(&q)?.transpose(1, 2)?;
    let k = k_norm.forward(&k_proj.forward(&input)?.reshape([1i64, SEQ, KV, D])?)?.transpose(1, 2)?;
    let v = v_proj.forward(&input)?.reshape([1i64, SEQ, KV, D])?.transpose(1, 2)?;

    let q = apply_rope(&q, &cos, &sin, 2)?;
    let k = apply_rope(&k, &cos, &sin, 2)?;
    let k = repeat_kv(&k, H / KV)?;
    let v = repeat_kv(&v, H / KV)?;

    // Causal scores + softmax + mix + gate + out.
    let mut mask = vec![0.0f32; (SEQ * SEQ) as usize];
    for i in 0..SEQ {
        for j in 0..SEQ {
            if j > i {
                mask[(i * SEQ + j) as usize] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from(mask).reshape([SEQ, SEQ])?.to(dev)?;
    let scores = q.matmul(k.transpose(-1, -2)?)? * (1.0 / (D as f32).sqrt()) + mask;
    let probs = scores.softmax([-1])?;
    let ctx = probs.matmul(v)?.transpose(1, 2)?.reshape([1i64, SEQ, H * D])?;
    let gated = ctx * gate.sigmoid();
    let out = o_proj.forward(gated)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&val, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((val - exp).abs() < 1e-3, "out[{i}] = {val}, expected {exp}");
    }
    Ok(())
}
