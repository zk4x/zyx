// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The qwen35 projection GEMMs with the PQ2_0 dequant fused into them, split by
//! phase: `prefill_gemm` is the prefill sequence length, `decode_gemm` is a
//! single token plus the lm_head.
//!
//! One bench entry per phase, all ten projections of the two block types and
//! the output projection: this is the graph the port builds (`QWeight.dequantize`
//! is part of the traced step), so the measured kernel is the one the model
//! runs. Weights are `[rows*cols/128, 34]` U8 super-blocks, the checkpoint's
//! own layout.

use zyx::{DType, Dev, QDType, Tensor, ZyxError};

use super::shapes::{
    CONV_DIM, FFN, HEADS, HEAD_DIM, HIDDEN, KV_HEADS, PQ2_BLOCK, PQ2_ROW, SEQ, VALUE_DIM, VOCAB,
};
use crate::bench::{bench_dev, run};

/// `(name, k, n)` of every projection, in the order a block runs them: the
/// full-attention block's four, the gated-delta block's three, then the SwiGLU
/// feed-forward the two share.
const PROJS: [(&str, i64, i64); 10] = [
    ("attn_q", HIDDEN, 2 * HEADS * HEAD_DIM),
    ("attn_k", HIDDEN, KV_HEADS * HEAD_DIM),
    ("attn_v", HIDDEN, KV_HEADS * HEAD_DIM),
    ("attn_output", HEADS * HEAD_DIM, HIDDEN),
    ("attn_qkv", HIDDEN, CONV_DIM),
    ("attn_gate", HIDDEN, VALUE_DIM),
    ("ssm_out", VALUE_DIM, HIDDEN),
    ("ffn_gate", HIDDEN, FFN),
    ("ffn_up", HIDDEN, FFN),
    ("ffn_down", FFN, HIDDEN),
];

/// Prefill GEMMs: every projection at the padded prefill width.
pub fn prefill_gemm() -> Result<(), ZyxError> {
    gemms(&[SEQ])
}

/// Decode GEMMs: every projection at a single token, plus the lm_head.
pub fn decode_gemm() -> Result<(), ZyxError> {
    gemms(&[1])?;
    lm_head()
}

/// A PQ2_0 weight on `dev`: raw `[rows*cols/128, 34]` U8 super-blocks, exactly
/// the shape the checkpoint stores and `QWeight.dequantize` consumes.
fn pq2(dev: Dev, rows: i64, cols: i64) -> Result<Tensor, ZyxError> {
    Tensor::rand([rows * cols / PQ2_BLOCK, PQ2_ROW], DType::U8)?.to(dev)
}

fn gemms(sizes: &[i64]) -> Result<(), ZyxError> {
    let dev = bench_dev();
    for &seq in sizes {
        for (name, k, n) in PROJS {
            eprintln!("[mark] {name}: rand");
            let x = Tensor::rand([seq, k], DType::F16)?.to(dev)?;
            let w = pq2(dev, n, k)?;
            eprintln!("[mark] {name}: built");
            println!("qwen matmul {name} [{seq}x{k}]x[{k}x{n}]");
            let w = w.dequantize(QDType::PQ2_0, DType::F16)?.reshape([n, k])?;
            run(x.dot_dtype_output(w.t(), DType::F32, DType::F32)?)?;
            eprintln!("[mark] {name}: ran");
        }
    }
    Ok(())
}

/// lm_head only ever runs at decode length: one token, the whole table.
fn lm_head() -> Result<(), ZyxError> {
    let dev = bench_dev();
    let x = Tensor::rand([1i64, HIDDEN], DType::F16)?.to(dev)?;
    let w = pq2(dev, VOCAB, HIDDEN)?;
    println!("qwen matmul lm_head [1x{HIDDEN}]x[{HIDDEN}x{VOCAB}]");
    let w = w
        .dequantize(QDType::PQ2_0, DType::F16)?
        .reshape([VOCAB, HIDDEN])?;
    run(x.dot_dtype_output(w.t(), DType::F32, DType::F32)?)?;
    Ok(())
}
