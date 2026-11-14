// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Greedy token parity with llama.cpp.
//!
//! llama.cpp reference (`llama-server`, same checkpoint, `temperature=0`,
//! `top_k=1`, raw prompt `Hello`, no chat template): the first five generated
//! token ids are `[11, 353, 2688, 264, 5286]` (", I'm a student").

use bonsai_27b::Bonsai;
use zyx::{DType, Tensor, ZyxError};

const MODEL: &str = "/home/x/Dev/cpp/llama.cpp/models/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const TOKENIZER: &str = "/home/x/Dev/rust/zyx/examples/models/qwen3.8-27b/tokenizer.json";

fn argmax(row: &[f32]) -> i64 {
    let mut best = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best as i64
}

#[test]
fn first_five_tokens() -> Result<(), ZyxError> {
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER).unwrap();
    let prompt: Vec<i64> = tokenizer
        .encode("Hello", true)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&id| i64::from(id))
        .collect();
    assert_eq!(prompt, vec![9419], "tokenizer prompt ids");

    let mut model = Bonsai::from_gguf(MODEL, 8192, 2048)?;
    let input = Tensor::from_vec(prompt, [1i64])?;
    let logits = model.forward(&input, 0)?;
    let mut row = logits.squeeze([0]).cast(DType::F32).to_vec::<f32>()?;
    let mut pos = 1i64;
    let mut generated = Vec::new();
    for _ in 0..5 {
        let next = argmax(&row);
        generated.push(next);
        let input = Tensor::from_vec(vec![next], [1i64])?;
        let logits = model.forward(&input, pos)?;
        row = logits.squeeze([0]).cast(DType::F32).to_vec::<f32>()?;
        pos += 1;
    }
    assert_eq!(generated, vec![11, 353, 2688, 264, 5286], "first five tokens vs llama.cpp");
    Ok(())
}
