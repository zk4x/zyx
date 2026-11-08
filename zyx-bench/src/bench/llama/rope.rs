// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama RoPE over a `[1, heads, seq, head_dim]` tensor.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{CTX, HEAD_DIM, HEADS, SIZES};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let half = HEAD_DIM / 2;
    // RoPE tables: [max_context, head_dim/2], as `precompute_rope_freqs` makes.
    let cos = Tensor::rand([CTX, half], DType::F16)?.to(dev)?;
    let sin = Tensor::rand([CTX, half], DType::F16)?.to(dev)?;
    for seq in SIZES {
        let x = Tensor::rand([1, HEADS, seq, HEAD_DIM], DType::F16)?.to(dev)?;
        println!("llama rope [1x{HEADS}x{seq}x{HEAD_DIM}]");
        let c = cos.narrow(0, 0i64, seq)?;
        let s = sin.narrow(0, 0i64, seq)?;
        run(x.rope(c, s)?)?;
    }
    Ok(())
}
