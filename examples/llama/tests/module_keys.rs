// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `#[derive(Module)]` on `Llama`: the `Vec<DecoderBlock>` stack is walked
//! element by element (torch-style labels, `layers.{i}.…`), and `plans` is
//! skipped because it is `#[no_param]`.
//!
//! Runs on whatever backend `ZYX_DEV` selects; it allocates weights and builds
//! the tape, and launches no kernel.

use llama::{Llama, LlamaConfig};
use zyx::{DType, Module};

fn cfg() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        max_position_embeddings: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        head_dim: Some(4),
        rope_scaling: None,
        tie_word_embeddings: false,
        max_context: 8,
    }
}

#[test]
fn llama_module_walks_layers_and_skips_plans() {
    let model = Llama::random(&cfg(), DType::F32);
    let labels: Vec<String> = model.iter_tensors().map(|(label, _)| label).collect();

    // One entry per tensor: `iter()` and `iter_tensors()` must agree.
    assert_eq!(labels.len(), model.iter().count(), "labels: {labels:?}");

    for expected in [
        "embedding.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.cache_k",
        "layers.0.self_attn.cache_v",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.1.self_attn.o_proj.weight",
        "layers.1.input_layernorm.weight",
        "norm.weight",
        "lm_head.weight",
    ] {
        assert!(labels.iter().any(|label| label == expected), "missing {expected}: {labels:?}");
    }

    // `arange` is an index buffer, not a parameter.
    assert!(!labels.iter().any(|label| label.ends_with(".arange")), "arange promoted: {labels:?}");
    // `plans` is traced state, never part of the module tree.
    assert!(!labels.iter().any(|label| label.starts_with("plans")), "plans walked: {labels:?}");

    let mut unique = labels.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), labels.len(), "duplicate labels: {labels:?}");

    // The tape is built from the module tree alone.
    let tape = zyx::Tape::new(&model).unwrap();
    tape.add(&zyx::Tensor::zeros([1, 1], DType::I64)).unwrap();
}

#[test]
fn llama_forward_runs_from_the_derived_module_tree() {
    let model_cfg = cfg();
    let vocab = model_cfg.vocab_size as i64;
    let mut model = Llama::random(&model_cfg, DType::F32);
    let ids = zyx::Tensor::from_vec(vec![1i64, 2, 3], [1, 3]).unwrap();
    let logits = model.forward(&ids, 0).cast(DType::F32);
    let values: Vec<f32> = logits.try_into().unwrap();
    assert_eq!(values.len() as i64, vocab, "[batch, vocab] logits for the last position");
    assert!(values.iter().all(|value| value.is_finite()), "{values:?}");
}
