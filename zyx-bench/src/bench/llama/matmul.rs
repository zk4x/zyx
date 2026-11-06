// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Every projection of a llama-3.2-3B decoder block, at decode and prefill.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, SIZES, VOCAB};
use crate::bench::{bench_dev, run};

pub fn bench() -> Result<(), ZyxError> {
    let dev = bench_dev();
    // (name, k, n) of every projection, in the order a block runs them.
    let projs = [
        ("q_proj", HIDDEN, HEADS * HEAD_DIM),
        ("k_proj", HIDDEN, KV_HEADS * HEAD_DIM),
        ("v_proj", HIDDEN, KV_HEADS * HEAD_DIM),
        ("o_proj", HEADS * HEAD_DIM, HIDDEN),
        ("gate_proj", HIDDEN, INTERMEDIATE),
        ("up_proj", HIDDEN, INTERMEDIATE),
        ("down_proj", INTERMEDIATE, HIDDEN),
    ];
    for seq in SIZES {
        for (name, k, n) in projs {
            let x = Tensor::rand([seq, k], DType::F16)?.to(dev)?;
            let w = Tensor::rand([n, k], DType::F16)?.to(dev)?;
            println!("llama matmul {name} [{seq}x{k}]x[{k}x{n}]");
            run(x.matmul(w.t())?)?;
        }
    }
    // lm_head only ever runs at decode length.
    let x = Tensor::rand([1i64, HIDDEN], DType::F16)?.to(dev)?;
    let w = Tensor::rand([VOCAB, HIDDEN], DType::F16)?.to(dev)?;
    println!("llama matmul lm_head [1x{HIDDEN}]x[{HIDDEN}x{VOCAB}]");
    run(x.matmul(w.t())?)?;
    Ok(())
}
