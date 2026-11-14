// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! GGUF load check for the Ternary-Bonsai-2-27B file: metadata accessors,
//! qwen35 geometry, Prism manifest and the two new loader dtypes (PQ2_0 and
//! BF16) all read back in one pass.

use zyx::{DType, Tensor, ZyxError};

const MODEL: &str = "/home/x/Dev/cpp/llama.cpp/models/Ternary-Bonsai-2-27B-PQ2_0.gguf";

#[test]
fn load_bonsai_gguf() -> Result<(), ZyxError> {
    let (meta, tensors) = Tensor::load_gguf(MODEL)?;
    assert_eq!(meta.int("qwen35.block_count")?, 64);
    assert_eq!(meta.int("qwen35.embedding_length")?, 5120);
    assert_eq!(meta.int("qwen35.feed_forward_length")?, 17408);
    assert_eq!(meta.int("qwen35.attention.head_count")?, 24);
    assert_eq!(meta.int("qwen35.attention.head_count_kv")?, 4);
    assert_eq!(meta.int("qwen35.rope.dimension_count")?, 64);
    assert_eq!(meta.float("qwen35.rope.freq_base")?, 10_000_000.0);
    assert_eq!(meta.opt_int("qwen35.nextn_predict_layers")?, None);
    assert!(meta.bool("prism.hadamard.gdn_v_grouped")?);
    assert_eq!(meta.int("prism.hadamard.block_size")?, 1024);
    assert_eq!(meta.strings("prism.hadamard.weight_names")?.len(), 401);
    assert_eq!(meta.strings("prism.hadamard.inverse_weight_names")?, vec!["token_embd.weight"]);
    assert_eq!(meta.ints("prism.hadamard.sign_widths")?, vec![5120, 6144, 17408]);
    assert_eq!(meta.ints("prism.hadamard.sign_values")?.len(), 28672);

    let emb = &tensors["token_embd.weight"];
    assert_eq!(emb.dtype(), DType::U8);
    assert_eq!(emb.resolve_shape(), vec![248_320 * 40, 34]);
    let output = &tensors["output.weight"];
    assert_eq!(output.resolve_shape(), vec![248_320 * 40, 34]);
    let beta = &tensors["blk.0.ssm_beta.weight"];
    assert_eq!(beta.dtype(), DType::BF16);
    assert_eq!(beta.resolve_shape(), vec![48, 5120]);
    assert_eq!(tensors["blk.0.attn_qkv.weight"].resolve_shape(), vec![10_240 * 40, 34]);
    assert_eq!(tensors["blk.3.attn_q.weight"].resolve_shape(), vec![12_288 * 40, 34]);
    assert_eq!(tensors["blk.0.ssm_conv1d.weight"].resolve_shape(), vec![10_240, 4]);
    Ok(())
}
