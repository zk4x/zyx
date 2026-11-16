// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Llama-3.x model definition and weight loading.
//!
//! The model lives outside the `zyx`/`zyx-nn` libraries on purpose (the same
//! split tinygrad uses with `extra/models/llama.py`): the CLI in `main.rs` and
//! benchmarks elsewhere depend on this crate, but the generic tensor library
//! never gains a specific model.
//!
//! All the reusable machinery — the pre-norm decoder block with grouped-query
//! attention, RoPE, a kv-cache and the SwiGLU feed-forward — lives in `zyx-nn`.
//! What is left here is llama's assembly: config, the decoder stack and the
//! tape. The fields carry the GGUF tensor names, so the checkpoint binds
//! through [`Module::set_params`] with no renaming step.

use std::collections::HashMap;
use std::path::Path;

use zyx::{DType, Dev, FrozenTape, Module, Tape, Tensor, ZyxError};
use zyx_derive::{gguf, Module};
use zyx_nn::{DecoderBlock, Embedding, Linear, RMSNorm, RopeScaling};

/// The device the weights are moved to once, at load. Leaving them in the
/// mmap'd disk pool makes the frozen plan stage every weight again on every
/// replay (a disk->device copy node per tensor per step), which is ~2 s per
/// token against ~32 ms of kernels. `ZYX_LLAMA_DEVICE` overrides the pick
/// (e.g. `cuda:0`, `c`, `vulkan:0`); by default the first available non-CPU
/// device is used.
fn compute_device() -> Dev {
    Dev::all()
        .into_iter()
        .find(|d| !matches!(d, Dev::C))
        .unwrap_or(Dev::C)
}

/// Model hyperparameters, built from the checkpoint's own GGUF metadata.
///
/// Every field is read from `<general.architecture>.<suffix>` by
/// `#[derive(gguf)]`, and `default = ...` on each field feeds the generated
/// `impl Default` (llama-3.2 values). `tie_word_embeddings` and `max_context`
/// are not in the metadata: the head is tied exactly when the file has no
/// `output.weight`, and the cache length is a runtime choice, so both come
/// from the caller.
#[derive(Debug, Clone, gguf)]
pub struct LlamaConfig {
    /// Embedding table size.
    #[gguf("vocab_size", default = 128256)]
    pub vocab_size: i64,
    /// Residual stream width.
    #[gguf("embedding_length", default = 3072)]
    pub hidden_size: i64,
    /// SwiGLU inner width.
    #[gguf("feed_forward_length", default = 8192)]
    pub intermediate_size: i64,
    /// Number of decoder layers.
    #[gguf("block_count", default = 28)]
    pub num_hidden_layers: i64,
    /// Number of query heads.
    #[gguf("attention.head_count", default = 24)]
    pub num_attention_heads: i64,
    /// Number of key/value heads (GQA); `None` means equal to the query heads.
    #[gguf("attention.head_count_kv", default = Some(8))]
    pub num_key_value_heads: Option<i64>,
    /// RoPE table length.
    #[gguf("context_length", default = 131072)]
    pub max_position_embeddings: i64,
    /// RMSNorm epsilon.
    #[gguf("attention.layer_norm_rms_epsilon", default = 1e-5)]
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    #[gguf("rope.freq_base", default = 500000.0)]
    pub rope_theta: f64,
    /// Per-head width; `None` derives it from the hidden size.
    #[gguf("attention.key_length", default = Some(128))]
    pub head_dim: Option<i64>,
    /// Optional RoPE scaling block, present when the file declares its type.
    #[gguf(group = "rope.scaling", gate = "rope.scaling.type", default = None)]
    pub rope_scaling: Option<RopeScaling>,
    /// Whether the lm head reuses the embedding matrix. llama-3.2's checkpoint
    /// carries no `output.weight`, so the default is the tied head that
    /// [`Llama::from_gguf`] derives from the file.
    #[gguf(skip, default = true)]
    pub tie_word_embeddings: bool,
    /// kv-cache length actually allocated.
    #[gguf(skip, default = 4096)]
    pub max_context: usize,
}

impl LlamaConfig {
    /// Key/value head count, defaulting to the query head count (no GQA).
    #[must_use]
    pub fn num_key_value_heads(&self) -> i64 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head width, defaulting to `hidden_size / num_attention_heads`.
    #[must_use]
    pub fn head_dim(&self) -> i64 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// A llama decoder stack with a preallocated kv-cache.
///
/// Every field carries the GGUF tensor name it binds to (`token_embd.weight`,
/// `blk.{i}.attn_q.weight`, `output_norm.weight`, `output.weight`), so
/// `Module::set_params` loads a checkpoint with no renaming.
#[derive(Module)]
pub struct Llama {
    token_embd: Embedding,
    blk: Vec<DecoderBlock>,
    output_norm: RMSNorm,
    output: Linear,
    /// Traced steps, one per sequence length, replayed afterwards.
    ///
    /// Tracing re-kernelizes hundreds of kernels per call, which dominated the
    /// end-to-end time (~2 s per step against ~32 ms of GPU work at decode).
    /// The token buffer and the two position variables are rebound per replay.
    #[no_param]
    plans: HashMap<i64, FrozenTape>,
}

impl Llama {
    /// Builds the model's skeleton: every tensor at `dtype` with the shape the
    /// checkpoint will have, the kv-caches zeroed.
    ///
    /// Load weights with [`Module::set_params`], keyed by the GGUF names.
    pub fn new(cfg: &LlamaConfig, dtype: DType) -> Self {
        let token_embd = Embedding::new(cfg.vocab_size, cfg.hidden_size, dtype).unwrap();
        let output = Linear {
            weight: if cfg.tie_word_embeddings {
                token_embd.weight.clone()
            } else {
                Linear::new(cfg.hidden_size, cfg.vocab_size, false, dtype)
                    .unwrap()
                    .weight
            },
            bias: None,
        };

        let mut blk = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for _ in 0..cfg.num_hidden_layers {
            blk.push(
                DecoderBlock::new(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads(),
                    cfg.head_dim(),
                    cfg.max_position_embeddings,
                    cfg.rope_theta,
                    cfg.rope_scaling,
                    cfg.max_context as i64,
                    cfg.rms_norm_eps,
                    dtype,
                )
                .unwrap(),
            );
        }

        let mut output_norm = RMSNorm::new(cfg.hidden_size, dtype);
        output_norm.eps = cfg.rms_norm_eps;
        Self {
            token_embd,
            blk,
            output_norm,
            output,
            plans: std::collections::HashMap::new(),
        }
    }

    /// Loads a GGUF checkpoint and binds it into a skeleton keyed by the file's
    /// own tensor names.
    ///
    /// # Errors
    ///
    /// Errors if the file is not a readable GGUF, if a required metadata key
    /// is missing, or if a per-layer metadata value is not uniform (the model
    /// has one value for all layers).
    pub fn from_gguf(
        path: impl AsRef<Path>,
        dtype: DType,
        max_context: usize,
    ) -> Result<Self, ZyxError> {
        let (metadata, mut tensors) = Tensor::load_gguf(path)?;
        // A tied checkpoint has no separate output projection.
        let tie_word_embeddings = !tensors.contains_key("output.weight");
        let cfg = LlamaConfig::from_gguf(&metadata, tie_word_embeddings, max_context)?;
        // The GGUF is mmap'd, so every weight starts in the disk pool. The
        // plan stages a disk-pool value with a copy node, and replay runs
        // that copy again on every step: the first trace measured ~8.8 GB of
        // weight uploads per token, ~2 s of the 2.05 s step, against ~32 ms of
        // kernels. Moving the weights once here (the same fix as the MNIST
        // example's dataset) leaves the plan with device-resident leaves.
        let dev = compute_device();
        for weight in tensors.values_mut() {
            *weight = weight.to(dev)?;
        }
        let mut model = Self::new(&cfg, dtype);
        model.set_params(&mut tensors);
        if cfg.tie_word_embeddings {
            model.output.weight = model.token_embd.weight.clone();
        }
        // llama.cpp runs the norms on the f32 residual stream whatever the
        // matmul dtype is; the GGUF already stores the norm weights f32.
        model.output_norm.weight = model.output_norm.weight.cast(DType::F32);
        for block in &mut model.blk {
            block.attn_norm.weight = block.attn_norm.weight.cast(DType::F32);
            block.ffn_norm.weight = block.ffn_norm.weight.cast(DType::F32);
        }
        Ok(model)
    }

    /// Every tensor gets the shape the GGUF checkpoint has, at `dtype`, filled
    /// with random values, so the resulting graph is identical to a real
    /// forward pass.
    #[must_use]
    pub fn random(cfg: &LlamaConfig, dtype: DType) -> Self {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim();
        let kv_dim = cfg.num_key_value_heads() * cfg.head_dim();
        let rand = |shape: &[i64]| Tensor::rand(shape.iter().copied(), dtype).unwrap();

        let mut vb = HashMap::new();
        vb.insert("token_embd.weight".to_string(), rand(&[vocab, hidden]));
        if !cfg.tie_word_embeddings {
            vb.insert("output.weight".to_string(), rand(&[vocab, hidden]));
        }
        vb.insert("output_norm.weight".to_string(), rand(&[hidden]));
        for i in 0..cfg.num_hidden_layers {
            let key = |suffix: &str| format!("blk.{i}.{suffix}");
            vb.insert(key("attn_q.weight"), rand(&[q_dim, hidden]));
            vb.insert(key("attn_k.weight"), rand(&[kv_dim, hidden]));
            vb.insert(key("attn_v.weight"), rand(&[kv_dim, hidden]));
            vb.insert(key("attn_output.weight"), rand(&[hidden, q_dim]));
            vb.insert(key("ffn_gate.weight"), rand(&[intermediate, hidden]));
            vb.insert(key("ffn_up.weight"), rand(&[intermediate, hidden]));
            vb.insert(key("ffn_down.weight"), rand(&[hidden, intermediate]));
            vb.insert(key("attn_norm.weight"), rand(&[hidden]));
            vb.insert(key("ffn_norm.weight"), rand(&[hidden]));
        }
        let mut model = Self::new(cfg, dtype);
        model.set_params(&mut vb);
        if cfg.tie_word_embeddings {
            model.output.weight = model.token_embd.weight.clone();
        }
        model
    }

    /// Runs one forward pass and realizes it.
    ///
    /// `input_ids` is `[batch, seq_len]`; `start_pos` is the absolute position
    /// of the first token in the kv-cache. Returns the logits for the last
    /// position, `[batch, vocab]`.
    ///
    /// The first call for a given sequence length traces the step and freezes
    /// it; every later call of that length replays the plan, so no
    /// kernelization, hashing or search runs per token. The prefill and the
    /// single-token decode step are separate plans because `Embedding`'s shape
    /// arithmetic turns the sequence dim into a graph value, not a symbolic
    /// one.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Tensor {
        let [_b_size, seq_len] = input_ids.dims::<2>().unwrap();
        let seq = seq_len.item::<i64>();
        let pos = Tensor::variable(start_pos as i64);
        let cache_len = Tensor::variable(start_pos as i64 + seq);
        if let Some(plan) = self.plans.get(&seq) {
            let out = plan.replay([input_ids, &pos, &cache_len]).unwrap();
            return out.into_iter().next().unwrap();
        }

        // One pass over the module tree covers every parameter and buffer,
        // including the kv-caches the attention block owns.
        let tape = Tape::new(&*self).unwrap();
        // The token buffer is a declared input: replay rebinds it, which is
        // what makes a step repeatable without retracing.
        tape.add(input_ids).unwrap();

        let mut xs = self.token_embd.forward(input_ids).unwrap().cast(DType::F32);
        for block in self.blk.iter_mut() {
            xs = block.forward(&xs, &pos, &cache_len).unwrap();
        }
        xs = self.output_norm.forward(xs).unwrap();
        xs = xs.narrow(1, seq_len - 1i64, 1i64).unwrap().squeeze([1]);
        let out = self.output.forward(xs).unwrap();

        let mut outputs: Vec<Tensor> = vec![out];
        for block in &self.blk {
            outputs.push(block.cache_k.clone());
            outputs.push(block.cache_v.clone());
        }
        let frozen = tape
            .freeze([input_ids, &pos, &cache_len], &outputs)
            .unwrap();
        // Replay immediately: the trace above only built the graph, and this
        // call still has to produce its logits.
        let res = frozen.replay([input_ids, &pos, &cache_len]).unwrap();
        self.plans.insert(seq, frozen);
        res.into_iter().next().unwrap()
    }
}
