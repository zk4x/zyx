// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama GQA attention: `softmax(q @ k^T * scale) @ v` with the kv repeat.

use std::ops::Mul;

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HEAD_DIM, HEADS, KV_HEADS, SCALE, SEQ, repeat_kv};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let scale = Tensor::from(SCALE).cast(DType::F16).to(dev)?;
    let n_rep = HEADS / KV_HEADS;
    for (seq, cache) in [(1i64, 1i64), (1, SEQ), (SEQ, SEQ)] {
        let q = Tensor::rand([1, HEADS, seq, HEAD_DIM], DType::F16)?.to(dev)?;
        let k = Tensor::rand([1, KV_HEADS, cache, HEAD_DIM], DType::F16)?.to(dev)?;
        let v = Tensor::rand([1, KV_HEADS, cache, HEAD_DIM], DType::F16)?.to(dev)?;
        println!("llama attention q[1x{HEADS}x{seq}x{HEAD_DIM}] kv[1x{KV_HEADS}x{cache}x{HEAD_DIM}]");
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;
        let scores = q.matmul(k.transpose(2, 3)?)?.mul(&scale);
        let pr = scores.softmax([-1])?;
        run(pr.matmul(v)?)?;
    }
    Ok(())
}
