// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Llama-3.x model definition and weight loading.
//!
//! The model lives outside the `zyx`/`zyx-nn` libraries on purpose (the same
//! split tinygrad uses with `extra/models/llama.py`): the CLI in `main.rs` and
//! benchmarks elsewhere depend on this crate, but the generic tensor library
//! never gains a specific model.
//!
//! All the reusable machinery — grouped-query attention with RoPE and a
//! kv-cache, the SwiGLU feed-forward, the pre-norm decoder block — lives in
//! `zyx-nn`. What is left here is llama's assembly: config, checkpoint naming,
//! and the tape.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use zyx::{DType, GGUFMetadataValue, Module, Tape, Tensor, ZyxError};
use zyx_nn::{DecoderBlock, Embedding, GqaAttention, Linear, RMSNorm, RopeConfig, RopeScaling as NnRopeScaling, SwiGlu};

/// Llama-3.1 style RoPE frequency scaling.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Scale applied to the low-frequency end.
    pub factor: f64,
    /// Upper wavelength bound of the interpolated band.
    pub high_freq_factor: f64,
    /// Lower wavelength bound of the interpolated band.
    pub low_freq_factor: f64,
    /// Context length the original RoPE table was built for.
    pub original_max_position_embeddings: usize,
    /// Scaling variant name (`llama3` today); recorded, not branched on.
    pub rope_type: String,
}

/// Model hyperparameters, deserialized straight from a Hugging Face
/// `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Embedding table size.
    pub vocab_size: usize,
    /// Residual stream width.
    pub hidden_size: usize,
    /// SwiGLU inner width.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA); `None` means equal to the query heads.
    pub num_key_value_heads: Option<usize>,
    /// RoPE table length.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// Per-head width; `None` derives it from the hidden size.
    pub head_dim: Option<usize>,
    /// Optional RoPE scaling block.
    pub rope_scaling: Option<RopeScaling>,
    /// Whether the lm head reuses the embedding matrix.
    pub tie_word_embeddings: bool,
    /// kv-cache length actually allocated.
    #[serde(default = "default_max_context")]
    pub max_context: usize,
}

#[derive(Deserialize)]
struct ShardedIndex {
    weight_map: HashMap<String, String>,
}

const fn default_max_context() -> usize {
    4096
}

impl LlamaConfig {
    /// Key/value head count, defaulting to the query head count (no GQA).
    #[must_use]
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head width, defaulting to `hidden_size / num_attention_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn rope(&self) -> RopeConfig {
        RopeConfig {
            head_dim: self.head_dim() as i64,
            max_position_embeddings: self.max_position_embeddings as i64,
            theta: self.rope_theta,
            scaling: self.rope_scaling.as_ref().map(|s| NnRopeScaling {
                factor: s.factor,
                high_freq_factor: s.high_freq_factor,
                low_freq_factor: s.low_freq_factor,
                original_max_position_embeddings: s.original_max_position_embeddings as i64,
            }),
        }
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: Some(128),
            rope_scaling: None,
            tie_word_embeddings: false,
            max_context: 4096,
        }
    }
}

/// Reads a required GGUF metadata value.
fn gguf_value<'a>(metadata: &'a HashMap<String, GGUFMetadataValue>, key: &str) -> Result<&'a GGUFMetadataValue, ZyxError> {
    metadata
        .get(key)
        .ok_or_else(|| ZyxError::parse_error(format!("GGUF metadata is missing `{key}`").into()))
}

/// One integer metadata value, or the uniform per-layer value of an array.
fn gguf_int(metadata: &HashMap<String, GGUFMetadataValue>, key: &str) -> Result<i64, ZyxError> {
    fn one(value: &GGUFMetadataValue, key: &str) -> Result<i64, ZyxError> {
        match *value {
            GGUFMetadataValue::Uint8(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Int8(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Uint16(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Int16(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Uint32(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Int32(v) => Ok(i64::from(v)),
            GGUFMetadataValue::Uint64(v) => i64::try_from(v)
                .map_err(|_| ZyxError::parse_error(format!("GGUF metadata `{key}` does not fit i64").into())),
            GGUFMetadataValue::Int64(v) => Ok(v),
            _ => Err(ZyxError::parse_error(format!("GGUF metadata `{key}` is not an integer").into())),
        }
    }
    match gguf_value(metadata, key)? {
        GGUFMetadataValue::Array(items) => {
            let Some(first) = items.first() else {
                return Err(ZyxError::parse_error(format!("GGUF metadata `{key}` is an empty array").into()));
            };
            let value = one(first, key)?;
            for item in items.iter() {
                if one(item, key)? != value {
                    return Err(ZyxError::parse_error(
                        format!("GGUF metadata `{key}` has per-layer values; zyx models one value for all layers").into(),
                    ));
                }
            }
            Ok(value)
        }
        value => one(value, key),
    }
}

fn gguf_usize(metadata: &HashMap<String, GGUFMetadataValue>, key: &str) -> Result<usize, ZyxError> {
    usize::try_from(gguf_int(metadata, key)?)
        .map_err(|_| ZyxError::parse_error(format!("GGUF metadata `{key}` is negative").into()))
}

fn gguf_opt_usize(metadata: &HashMap<String, GGUFMetadataValue>, key: &str) -> Result<Option<usize>, ZyxError> {
    match metadata.get(key) {
        None => Ok(None),
        Some(_) => gguf_usize(metadata, key).map(Some),
    }
}

fn gguf_f64(metadata: &HashMap<String, GGUFMetadataValue>, key: &str) -> Result<f64, ZyxError> {
    match gguf_value(metadata, key)? {
        GGUFMetadataValue::Float32(v) => Ok(f64::from(*v)),
        GGUFMetadataValue::Float64(v) => Ok(*v),
        _ => Ok(gguf_int(metadata, key)? as f64),
    }
}

fn gguf_opt_str<'a>(metadata: &'a HashMap<String, GGUFMetadataValue>, key: &str) -> Result<Option<&'a str>, ZyxError> {
    match metadata.get(key) {
        None => Ok(None),
        Some(GGUFMetadataValue::String(value)) => Ok(Some(value)),
        Some(_) => Err(ZyxError::parse_error(format!("GGUF metadata `{key}` is not a string").into())),
    }
}

impl LlamaConfig {
    /// Builds the config from the checkpoint's own GGUF metadata, so the file
    /// is the single source of truth (no `config.json` beside it to disagree).
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, GGUFMetadataValue>,
        max_context: usize,
        tie_word_embeddings: bool,
    ) -> Result<Self, ZyxError> {
        let rope_scaling = match gguf_opt_str(metadata, "llama.rope.scaling.type")? {
            None => None,
            Some(rope_type) => Some(RopeScaling {
                factor: gguf_f64(metadata, "llama.rope.scaling.factor")?,
                high_freq_factor: gguf_f64(metadata, "llama.rope.scaling.high_freq_factor")?,
                low_freq_factor: gguf_f64(metadata, "llama.rope.scaling.low_freq_factor")?,
                original_max_position_embeddings: gguf_usize(metadata, "llama.rope.scaling.original_context_length")?,
                rope_type: rope_type.to_string(),
            }),
        };
        Ok(Self {
            vocab_size: gguf_usize(metadata, "llama.vocab_size")?,
            hidden_size: gguf_usize(metadata, "llama.embedding_length")?,
            intermediate_size: gguf_usize(metadata, "llama.feed_forward_length")?,
            num_hidden_layers: gguf_usize(metadata, "llama.block_count")?,
            num_attention_heads: gguf_usize(metadata, "llama.attention.head_count")?,
            num_key_value_heads: gguf_opt_usize(metadata, "llama.attention.head_count_kv")?,
            max_position_embeddings: gguf_usize(metadata, "llama.context_length")?,
            rms_norm_eps: gguf_f64(metadata, "llama.attention.layer_norm_rms_epsilon")?,
            rope_theta: gguf_f64(metadata, "llama.rope.freq_base")?,
            head_dim: gguf_opt_usize(metadata, "llama.attention.key_length")?,
            rope_scaling,
            tie_word_embeddings,
            max_context,
        })
    }
}

trait VarMap {
    fn remove_prefix(&mut self, prefix: &str) -> Self;
    fn take(&mut self, key: &str) -> Tensor;
}

impl VarMap for HashMap<String, Tensor> {
    fn remove_prefix(&mut self, prefix: &str) -> Self {
        let p = format!("{prefix}.");
        let mut res = HashMap::new();
        let paths: Vec<String> = self.keys().filter(|k| k.starts_with(&p)).cloned().collect();
        for path in paths {
            let t = self.remove(&path).unwrap();
            let stripped = path.strip_prefix(&p).unwrap().to_string();
            res.insert(stripped, t);
        }
        res
    }

    fn take(&mut self, key: &str) -> Tensor {
        self.remove(key)
            .unwrap_or_else(|| panic!("key {key} not found in weight map"))
    }
}

/// A llama decoder stack with a preallocated kv-cache.
pub struct Llama {
    embedding: Embedding,
    layers: Vec<DecoderBlock>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl Llama {
    /// Builds the model, consuming weights out of `vb` by key.
    ///
    /// Expected keys are the Hugging Face names: `model.embed_tokens.weight`,
    /// `model.layers.{i}.{...}`, `model.norm.weight`, and `lm_head.weight`
    /// (omitted when the head is tied to the embedding).
    pub fn new(cfg: &LlamaConfig, vb: &mut HashMap<String, Tensor>, dtype: DType) -> Self {
        let rope = cfg.rope();
        let embedding = Embedding::from_params(vb.take("model.embed_tokens.weight")).unwrap();
        let lm_head = Linear {
            weight: if cfg.tie_word_embeddings {
                embedding.weight.clone()
            } else {
                vb.take("lm_head.weight")
            },
            bias: None,
        };

        let mut vb_layers = vb.remove_prefix("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let mut layer_vb = vb_layers.remove_prefix(&i.to_string());
            // zyx-nn names an RMSNorm scale `scale`; checkpoints call it
            // `weight`. Rename here so `set_params` does the rest.
            for name in ["input_layernorm", "post_attention_layernorm"] {
                if let Some(t) = layer_vb.remove(&format!("{name}.weight")) {
                    layer_vb.insert(format!("{name}.scale"), t);
                }
            }
            let self_attn = GqaAttention::new(
                cfg.hidden_size as i64,
                cfg.num_attention_heads as i64,
                cfg.num_key_value_heads() as i64,
                &rope,
                cfg.max_context as i64,
                dtype,
            )
            .unwrap();
            let mlp = SwiGlu::new(cfg.hidden_size as i64, cfg.intermediate_size as i64, dtype).unwrap();
            let mut layer = DecoderBlock::new(self_attn, mlp, cfg.hidden_size as i64, cfg.rms_norm_eps, dtype);
            layer.set_params(&mut layer_vb);
            layers.push(layer);
        }

        let mut norm = RMSNorm::new(cfg.hidden_size as i64, dtype);
        norm.eps = cfg.rms_norm_eps;
        norm.scale = vb.take("model.norm.weight").cast(dtype);
        Self {
            embedding,
            layers,
            norm,
            lm_head,
        }
    }

    /// Builds a model with random weights, for benchmarks that want llama's
    /// shapes without loading a checkpoint.
    ///
    /// Loads a GGUF checkpoint and builds the model from the file alone: the
    /// weights, the hyperparameters and the tied-head flag all come out of the
    /// GGUF (no `config.json` needed beside it).
    ///
    /// # Errors
    ///
    /// Errors if the file is not a readable GGUF, if a required metadata key
    /// is missing, or if a per-layer metadata value is not uniform (the model
    /// has one value for all layers).
    pub fn from_gguf(path: impl AsRef<Path>, dtype: DType, max_context: usize) -> Result<Self, ZyxError> {
        let (metadata, tensors) = Tensor::load_gguf(path)?;
        // A tied checkpoint has no separate output projection.
        let tie_word_embeddings = !tensors.contains_key("output.weight");
        let cfg = LlamaConfig::from_gguf_metadata(&metadata, max_context, tie_word_embeddings)?;
        let mut vb = remap_gguf_weights(tensors);
        // GGUF keeps the norms in f32 while the compute dtype is f16/bf16;
        // the model has one dtype, so cast every parameter once at load.
        for tensor in vb.values_mut() {
            *tensor = tensor.cast(dtype);
        }
        Ok(Self::new(&cfg, &mut vb, dtype))
    }

    /// Every tensor gets the shape the Hugging Face checkpoint would have, at
    /// `dtype`, so the resulting graph is identical to a real forward pass.
    #[must_use]
    pub fn random(cfg: &LlamaConfig, dtype: DType) -> Self {
        let hidden = cfg.hidden_size as i64;
        let intermediate = cfg.intermediate_size as i64;
        let vocab = cfg.vocab_size as i64;
        let q_dim = (cfg.num_attention_heads * cfg.head_dim()) as i64;
        let kv_dim = (cfg.num_key_value_heads() * cfg.head_dim()) as i64;
        let rand = |shape: &[i64]| Tensor::rand(shape.iter().copied(), dtype).unwrap();

        let mut vb = HashMap::new();
        vb.insert("model.embed_tokens.weight".to_string(), rand(&[vocab, hidden]));
        if !cfg.tie_word_embeddings {
            vb.insert("lm_head.weight".to_string(), rand(&[vocab, hidden]));
        }
        vb.insert("model.norm.weight".to_string(), rand(&[hidden]));
        for i in 0..cfg.num_hidden_layers {
            let key = |suffix: &str| format!("model.layers.{i}.{suffix}");
            vb.insert(key("self_attn.q_proj.weight"), rand(&[q_dim, hidden]));
            vb.insert(key("self_attn.k_proj.weight"), rand(&[kv_dim, hidden]));
            vb.insert(key("self_attn.v_proj.weight"), rand(&[kv_dim, hidden]));
            vb.insert(key("self_attn.o_proj.weight"), rand(&[hidden, q_dim]));
            vb.insert(key("mlp.gate_proj.weight"), rand(&[intermediate, hidden]));
            vb.insert(key("mlp.up_proj.weight"), rand(&[intermediate, hidden]));
            vb.insert(key("mlp.down_proj.weight"), rand(&[hidden, intermediate]));
            vb.insert(key("input_layernorm.weight"), rand(&[hidden]));
            vb.insert(key("post_attention_layernorm.weight"), rand(&[hidden]));
        }
        Self::new(cfg, &mut vb, dtype)
    }

    /// Runs one forward pass and realizes it.
    ///
    /// `input_ids` is `[batch, seq_len]`; `start_pos` is the absolute position
    /// of the first token in the kv-cache. Returns the logits for the last
    /// position, `[batch, vocab]`.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Tensor {
        let [_b_size, seq_len] = input_ids.dims::<2>().unwrap();
        let tape = Tape::empty();
        tape.add(&self.embedding.weight).unwrap();
        // One pass over the module tree covers every parameter and buffer,
        // including the kv-caches the attention block owns.
        for layer in &self.layers {
            for (_, t) in layer.iter_tensors() {
                tape.add(t).unwrap();
            }
        }
        tape.add(&self.norm.scale).unwrap();
        tape.add(&self.lm_head.weight).unwrap();
        let mut xs = self.embedding.forward(input_ids).unwrap();
        // Symbolic positions: fresh variables each call, but identical kernel
        // IR every step (params hash by ordinal, not value) — one compile total.
        let pos = Tensor::variable(start_pos as i64);
        let cache_len = Tensor::variable(start_pos as i64 + seq_len.item::<i64>());
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &pos, &cache_len).unwrap();
        }
        xs = self.norm.forward(xs).unwrap();
        xs = xs.narrow(1, seq_len - 1i64, 1i64).unwrap().squeeze([1]);
        let out = self.lm_head.forward(xs).unwrap();
        let mut realize_args: Vec<&Tensor> = vec![&out];
        for layer in &self.layers {
            realize_args.push(&layer.self_attn.cache_k);
            realize_args.push(&layer.self_attn.cache_v);
        }
        tape.realize(realize_args).unwrap();
        out
    }
}

/// Rewrites llama.cpp GGUF tensor names to the Hugging Face names
/// [`Llama::new`] expects.
#[must_use]
pub fn remap_gguf_weights(mut gguf: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    if let Some(t) = gguf.remove("token_embd.weight") {
        out.insert("model.embed_tokens.weight".to_string(), t);
    }
    if let Some(t) = gguf.remove("output_norm.weight") {
        out.insert("model.norm.weight".to_string(), t);
    }
    if let Some(t) = gguf.remove("output.weight") {
        out.insert("lm_head.weight".to_string(), t);
    }
    // (GGUF suffix, HF suffix) — the only per-architecture table llama needs.
    const LAYER_MAP: &[(&str, &str)] = &[
        ("attn_q", "self_attn.q_proj.weight"),
        ("attn_k", "self_attn.k_proj.weight"),
        ("attn_v", "self_attn.v_proj.weight"),
        ("attn_output", "self_attn.o_proj.weight"),
        ("attn_norm", "input_layernorm.weight"),
        ("ffn_gate", "mlp.gate_proj.weight"),
        ("ffn_down", "mlp.down_proj.weight"),
        ("ffn_up", "mlp.up_proj.weight"),
        ("ffn_norm", "post_attention_layernorm.weight"),
    ];
    let layer_keys: Vec<String> = gguf.keys().filter(|k| k.starts_with("blk.")).cloned().collect();
    for key in layer_keys {
        let parts: Vec<&str> = key.splitn(4, '.').collect();
        if parts.len() < 4 {
            continue;
        }
        let layer: usize = parts[1].parse().unwrap_or(0);
        let Some((_, hf)) = LAYER_MAP.iter().find(|(gguf, _)| *gguf == parts[2]) else {
            continue;
        };
        let tensor = gguf.remove(&key).unwrap();
        out.insert(format!("model.layers.{layer}.{hf}"), tensor);
    }
    out
}

/// Loads weights from a `.safetensors` file, a `.gguf`, or a sharded index
/// `.json` pointing at safetensors shards.
#[must_use]
pub fn load_weights(weight_path: &str) -> HashMap<String, Tensor> {
    let path = std::path::Path::new(weight_path);
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "json" => {
            let content = std::fs::read_to_string(weight_path).unwrap();
            let index: ShardedIndex = serde_json::from_str(&content).unwrap();
            let base = path.parent().unwrap();
            let mut shard_files: Vec<&String> = index.weight_map.values().collect();
            shard_files.sort();
            shard_files.dedup();
            let mut all_weights = HashMap::new();
            for f in shard_files {
                let shard_path = base.join(f);
                let weights = Tensor::load(&shard_path).unwrap();
                all_weights.extend(weights);
            }
            all_weights
        }
        _ => Tensor::load(weight_path).unwrap(),
    }
}
