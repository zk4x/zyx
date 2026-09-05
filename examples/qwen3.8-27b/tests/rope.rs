// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Partial-RoPE application reference-side test (runs on CUDA).
//!
//! Golden: `examples/data/qwen3_8b_rope.safetensors` from `tests/rope_ref.py`.
//! Run the dump first: `cd tests && python3.12 rope_ref.py`.
//!
//! Applies rotation to the first `rot_dim` dims, passes the rest through:
//! out = cat(q_rot * cos + rotate_half(q_rot) * sin, q_pass).

use zyx::kernel::Dev;
use zyx::{Tensor, ZyxError};

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rot_dim: i64) -> Result<Tensor, ZyxError> {
    let last = x.rank() as i32 - 1;
    let head_dim: i64 = x.shape()[last as usize].item();
    let q_rot = x.narrow(last, 0i64, rot_dim)?;
    let q_pass = x.narrow(last, rot_dim, head_dim - rot_dim)?;
    let half = rot_dim / 2;
    let a = q_rot.narrow(last, 0i64, half)?;
    let b = q_rot.narrow(last, half, half)?;
    // rotate_half(x) = cat(-b, a)
    let neg_b = -&b;
    let rotated = Tensor::cat([&neg_b, &a], last)?;
    let out_rot = &q_rot * cos + &rotated * sin;
    Tensor::cat([&out_rot, &q_pass], last)
}

#[test]
fn rope() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_rope.safetensors")?;
    let q = goldens["q"].to(Dev::Cuda(0))?;
    let k = goldens["k"].to(Dev::Cuda(0))?;
    let cos = goldens["cos"].to(Dev::Cuda(0))?;
    let sin = goldens["sin"].to(Dev::Cuda(0))?;
    let expected_q = goldens["q_rot"].to_vec::<f32>()?;
    let expected_k = goldens["k_rot"].to_vec::<f32>()?;

    let out_q = apply_rope(&q, &cos, &sin, 4)?.to_vec::<f32>()?;
    let out_k = apply_rope(&k, &cos, &sin, 4)?.to_vec::<f32>()?;

    for (name, out, expected) in [("q", out_q, expected_q), ("k", out_k, expected_k)] {
        assert_eq!(out.len(), expected.len());
        for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-4, "{name}[{i}] = {v}, expected {e}");
        }
    }
    Ok(())
}
