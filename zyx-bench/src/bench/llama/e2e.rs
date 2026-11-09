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
    // Two passes: the first compiles/autotunes every kernel (and checks the
    // tokens), the second is the measured one. Timings are wall clock around
    // `forward` plus the result readback that forces it, so they include
    // everything a user pays per step.
    let mut got = Vec::with_capacity(EXPECTED.len());
    let mut prefill_ms = 0.0f64;
    let mut decode_ms: Vec<f64> = Vec::with_capacity(EXPECTED.len() - 1);
    for pass in 0..2 {
        let started = std::time::Instant::now();
        let mut last_ms = 0.0f64;
        let mut logits = model.forward(&prompt, 0);
        let mut pos = PROMPT.len();
        let mut tokens = Vec::with_capacity(EXPECTED.len());
        for step in 0..EXPECTED.len() {
            let probs: Vec<f32> = logits.clone().cast(DType::F32).try_into()?;
            // The readback above forced the step's kernels; the delta since the
            // previous readback is this step's wall time (prefill for step 0,
            // one token per decode step).
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            let step_ms = elapsed - last_ms;
            last_ms = elapsed;
            if pass == 1 {
                if step == 0 {
                    prefill_ms = step_ms;
                } else {
                    decode_ms.push(step_ms);
                }
            }
            // First maximum, exactly like llama.cpp's sampler at temperature 0.
            let mut next = 0usize;
            for (i, &x) in probs.iter().enumerate() {
                if x > probs[next] {
                    next = i;
                }
            }
            // Top-5 logits, the same shape `llama_ids` prints, so a token flip
            // can be classified as precision (same set, small gap) or as a bug.
            let mut idx: Vec<usize> = (0..probs.len()).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            print!("zyx step {step}:");
            for &i in &idx[..5] {
                print!(" {i}={:.6}", probs[i]);
            }
            println!();
            tokens.push(next as u32);
            // The last expected token needs no successor: stop after the 5th.
            if tokens.len() == EXPECTED.len() {
                break;
            }
            let token = Tensor::from(vec![next as u32]).reshape([1, 1])?;
            logits = model.forward(&token, pos);
            pos += 1;
        }
        got = tokens;
    }

    assert_eq!(got, EXPECTED, "greedy tokens differ from llama.cpp (f16 GGUF, temp 0)");
    let prompt_tokens = PROMPT.len() as f64;
    let decode_avg = decode_ms.iter().sum::<f64>() / decode_ms.len().max(1) as f64;
    println!(
        "llama e2e: {}/{} tokens match llama.cpp: {got:?}",
        got.len(),
        EXPECTED.len()
    );
    println!(
        "llama e2e: prefill {:.3} ms for {prompt_tokens:.0} tokens ({:.2} tok/s), decode {:.3} ms/token ({:.2} tok/s)",
        prefill_ms,
        prompt_tokens * 1e3 / prefill_ms,
        decode_avg,
        1e3 / decode_avg
    );
    Ok(())
}
