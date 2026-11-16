// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `#[derive(gguf)]`: every field reads `<general.architecture>.<suffix>`, a
//! group field is built from a key group behind a gate key, and skipped fields
//! come from the trailing constructor parameters.

use std::collections::HashMap;

use zyx::{GGUFMetadata, GGUFMetadataValue, ZyxError};
use zyx_derive::gguf;

#[derive(Debug, gguf)]
struct Scaling {
    #[gguf("factor")]
    factor: f64,
    #[gguf("high_freq_factor")]
    high_freq_factor: f64,
    #[gguf("low_freq_factor")]
    low_freq_factor: f64,
    #[gguf("original_context_length")]
    original_max_position_embeddings: i64,
}

#[derive(Debug, gguf)]
struct Config {
    #[gguf("embedding_length", default = 32)]
    dim: i64,
    #[gguf("attention.head_count_kv", default = Some(2))]
    kv_heads: Option<i64>,
    #[gguf("rope.freq_base", default = 10000.0)]
    theta: f64,
    #[gguf(group = "rope.scaling", gate = "rope.scaling.type", default = None)]
    scaling: Option<Scaling>,
    /// No attribute: the key is the field's own name.
    density: f64,
    #[gguf(skip, default = true)]
    tied: bool,
    #[gguf(skip, default = 4096)]
    max_context: i64,
}

fn metadata(arch: &str, entries: Vec<(&str, GGUFMetadataValue)>) -> GGUFMetadata {
    let mut values = HashMap::new();
    values.insert(
        "general.architecture".to_string(),
        GGUFMetadataValue::String(arch.to_string()),
    );
    for (key, value) in entries {
        values.insert(key.to_string(), value);
    }
    GGUFMetadata::new(values)
}

#[test]
fn gguf_derive_reads_arch_prefixed_keys() -> Result<(), ZyxError> {
    let meta = metadata(
        "llama",
        vec![
            ("llama.embedding_length", GGUFMetadataValue::Uint32(3072)),
            (
                "llama.attention.head_count_kv",
                GGUFMetadataValue::Uint32(8),
            ),
            ("llama.rope.freq_base", GGUFMetadataValue::Float32(500000.0)),
            (
                "llama.rope.scaling.type",
                GGUFMetadataValue::String("llama3".to_string()),
            ),
            (
                "llama.rope.scaling.factor",
                GGUFMetadataValue::Float32(32.0),
            ),
            (
                "llama.rope.scaling.high_freq_factor",
                GGUFMetadataValue::Float32(4.0),
            ),
            (
                "llama.rope.scaling.low_freq_factor",
                GGUFMetadataValue::Float32(1.0),
            ),
            (
                "llama.rope.scaling.original_context_length",
                GGUFMetadataValue::Uint32(8192),
            ),
            ("llama.density", GGUFMetadataValue::Float32(0.5)),
        ],
    );
    let cfg = Config::from_gguf(&meta, true, 4096)?;
    assert_eq!(cfg.dim, 3072);
    assert_eq!(cfg.kv_heads, Some(8));
    assert_eq!(cfg.theta, 500000.0);
    assert_eq!(cfg.density, 0.5);
    let scaling = cfg.scaling.expect("scaling present");
    assert_eq!(scaling.factor, 32.0);
    assert_eq!(scaling.high_freq_factor, 4.0);
    assert_eq!(scaling.low_freq_factor, 1.0);
    assert_eq!(scaling.original_max_position_embeddings, 8192);
    assert!(cfg.tied);
    assert_eq!(cfg.max_context, 4096);
    Ok(())
}

#[test]
fn gguf_derive_is_arch_agnostic_and_gates_groups() -> Result<(), ZyxError> {
    let meta = metadata(
        "qwen35",
        vec![
            ("qwen35.embedding_length", GGUFMetadataValue::Int64(512)),
            ("qwen35.rope.freq_base", GGUFMetadataValue::Float64(10000.0)),
            ("qwen35.density", GGUFMetadataValue::Float64(2.0)),
        ],
    );
    let cfg = Config::from_gguf(&meta, false, 128)?;
    assert_eq!(cfg.dim, 512);
    assert_eq!(cfg.kv_heads, None);
    assert_eq!(cfg.theta, 10000.0);
    assert_eq!(cfg.density, 2.0);
    assert!(
        cfg.scaling.is_none(),
        "no `qwen35.rope.scaling.type` gate key"
    );
    assert!(!cfg.tied);
    assert_eq!(cfg.max_context, 128);
    Ok(())
}

#[test]
fn gguf_derive_with_prefix_uses_the_prefix_verbatim() -> Result<(), ZyxError> {
    let meta = metadata(
        "llama",
        vec![
            ("custom.factor", GGUFMetadataValue::Float32(8.0)),
            ("custom.high_freq_factor", GGUFMetadataValue::Float32(3.0)),
            ("custom.low_freq_factor", GGUFMetadataValue::Float32(1.5)),
            (
                "custom.original_context_length",
                GGUFMetadataValue::Uint32(4096),
            ),
        ],
    );
    let scaling = Scaling::from_gguf_with_prefix(&meta, "custom.")?;
    assert_eq!(scaling.factor, 8.0);
    assert_eq!(scaling.high_freq_factor, 3.0);
    assert_eq!(scaling.low_freq_factor, 1.5);
    assert_eq!(scaling.original_max_position_embeddings, 4096);
    Ok(())
}

#[test]
fn gguf_derive_default_uses_the_declared_values() {
    let cfg = Config::default();
    assert_eq!(cfg.dim, 32);
    assert_eq!(cfg.kv_heads, Some(2));
    assert_eq!(cfg.theta, 10000.0);
    assert!(cfg.scaling.is_none(), "declared `default = None`");
    assert!(cfg.tied);
    assert_eq!(cfg.max_context, 4096);
    // No declared default: the field's own `Default::default()`.
    assert_eq!(cfg.density, 0.0);
    // A nested struct with no declared defaults still gets `Default`.
    let scaling = Scaling::default();
    assert_eq!(scaling.factor, 0.0);
    assert_eq!(scaling.high_freq_factor, 0.0);
    assert_eq!(scaling.low_freq_factor, 0.0);
    assert_eq!(scaling.original_max_position_embeddings, 0);
}
