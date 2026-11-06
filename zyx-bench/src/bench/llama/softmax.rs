// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama attention softmax over the cached axis.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HEADS, SEQ};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    // (query length, cached length): first token, a token against a full
    // prompt, and a full prefill block (causal mask, square scores).
    for (seq, cache) in [(1i64, 1i64), (1, SEQ), (SEQ, SEQ)] {
        let x = Tensor::rand([HEADS, seq, cache], DType::F16)?.to(dev)?;
        println!("llama softmax [{HEADS}x{seq}x{cache}]");
        run(x.softmax([-1])?)?;
    }
    Ok(())
}
