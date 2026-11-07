// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end llama correctness on the real f16 checkpoint.
//!
//! Loads `Llama-3.2-3B-Instruct-f16.gguf`, feeds a pinned prompt and decodes
//! greedily. The expected ids come from llama.cpp running the same GGUF on the
//! same card (`llama_ids`, `--temp 0`), so this is a reference, not a snapshot
//! of zyx's own output.
//!
//! On demand only: loading the checkpoint and compiling the 28 decoder layers
//! takes tens of seconds, which has no place in CI until it is much faster.

use llama::Llama;
use zyx::{DType, Tensor, ZyxError};

/// `"The capital of France is"` (BOS + 5 tokens) from the GGUF tokenizer.
const PROMPT: [u32; 6] = [128000, 791, 6864, 315, 9822, 374];

/// Greedy continuation from llama.cpp on the same f16 GGUF, GPU, temp 0:
/// `" Paris. The capital of"`.
const EXPECTED: [u32; 5] = [12366, 13, 578, 6864, 315];

pub fn e2e() -> Result<(), ZyxError> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/models/llama-3.2-3b");
    let mut model = Llama::from_gguf(dir.join("Llama-3.2-3B-Instruct-f16.gguf"), DType::F16, 4096)?;

    let prompt = Tensor::from(PROMPT.to_vec()).reshape([1, PROMPT.len() as i64])?;
    let mut logits = model.forward(&prompt, 0);
    let mut pos = PROMPT.len();
    let mut got = Vec::with_capacity(EXPECTED.len());
    for step in 0..EXPECTED.len() {
        let probs: Vec<f32> = logits.clone().cast(DType::F32).try_into()?;
        // First maximum, exactly like llama.cpp's sampler at temperature 0.
        let mut next = 0usize;
        for (i, &x) in probs.iter().enumerate() {
            if x > probs[next] {
                next = i;
            }
        }
        // Top-5 logits, the same shape `llama_ids` prints, so a token flip can
        // be classified as precision (same set, small gap) or as a bug.
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        print!("zyx step {step}:");
        for &i in &idx[..5] {
            print!(" {i}={:.6}", probs[i]);
        }
        println!();
        got.push(next as u32);
        // The last expected token needs no successor: stop after the 5th.
        if got.len() == EXPECTED.len() {
            break;
        }
        let token = Tensor::from(vec![next as u32]).reshape([1, 1])?;
        logits = model.forward(&token, pos);
        pos += 1;
    }

    assert_eq!(got, EXPECTED, "greedy tokens differ from llama.cpp (f16 GGUF, temp 0)");
    println!("llama e2e: {}/{} tokens match llama.cpp: {got:?}", got.len(), EXPECTED.len());
    Ok(())
}
