// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama kv-cache write: `cache[start..start+seq].assign(k_new)`.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{CTX, HEAD_DIM, KV_HEADS, SIZES};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    for seq in SIZES {
        let cache = Tensor::zeros([CTX, KV_HEADS, HEAD_DIM], DType::F16).to(dev)?;
        let k = Tensor::rand([seq, KV_HEADS, HEAD_DIM], DType::F16)?.to(dev)?;
        println!("llama kv_assign [{seq}x{KV_HEADS}x{HEAD_DIM}] into [{CTX}x..]");
        cache.narrow(0, 0, seq)?.assign(&k)?;
        // Reading the written window forces the store; the assign itself is
        // the kernel under test.
        run(cache.narrow(0, 0, seq)?)?;
    }
    Ok(())
}
