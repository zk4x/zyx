// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The llama-3.2-3B projection GEMMs, split by phase: `prefill_gemm` is the
//! prefill sequence length, `decode_gemm` is a single token plus the lm_head.

use zyx::{DType, Tensor, ZyxError};

use super::shapes::{HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, SEQ, VOCAB};
use crate::bench::{bench_dev, run};

/// `(name, k, n)` of every projection, in the order a block runs them.
const PROJS: [(&str, i64, i64); 7] = [
    ("q_proj", HIDDEN, HEADS * HEAD_DIM),
    ("k_proj", HIDDEN, KV_HEADS * HEAD_DIM),
    ("v_proj", HIDDEN, KV_HEADS * HEAD_DIM),
    ("o_proj", HEADS * HEAD_DIM, HIDDEN),
    ("gate_proj", HIDDEN, INTERMEDIATE),
    ("up_proj", HIDDEN, INTERMEDIATE),
    ("down_proj", INTERMEDIATE, HIDDEN),
];

/// Prefill GEMMs: every projection at the prefill sequence length.
pub fn prefill_gemm() -> Result<(), ZyxError> {
    gemms(&[SEQ])
}

/// Decode GEMMs: every projection at a single token, plus the lm_head.
pub fn decode_gemm() -> Result<(), ZyxError> {
    gemms(&[1])?;
    lm_head()
}

fn gemms(sizes: &[i64]) -> Result<(), ZyxError> {
    let dev = bench_dev();
    for &seq in sizes {
        for (name, k, n) in PROJS {
            eprintln!("[mark] {name}: rand");
            let x = Tensor::rand([seq, k], DType::F16)?.to(dev)?;
            let w = Tensor::rand([n, k], DType::F16)?.to(dev)?;
            eprintln!("[mark] {name}: built");
            println!("llama matmul {name} [{seq}x{k}]x[{k}x{n}]");
            run(x.matmul(w.t())?)?;
            eprintln!("[mark] {name}: ran");
        }
    }
    Ok(())
}

/// lm_head only ever runs at decode length.
fn lm_head() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let x = Tensor::rand([1i64, HIDDEN], DType::F16)?.to(dev)?;
    let w = Tensor::rand([VOCAB, HIDDEN], DType::F16)?.to(dev)?;
    println!("llama matmul lm_head [1x{HIDDEN}]x[{HIDDEN}x{VOCAB}]");
    run(x.matmul(w.t())?)?;
    Ok(())
}
