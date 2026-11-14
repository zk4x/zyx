// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! The Ternary-Bonsai-2-27B trunk: a qwen35 hybrid stack of 64 blocks, 16 of
//! them full-attention and 48 gated-delta-net (linear-attention) blocks.
//!
//! The forward is ordinary tensor arithmetic on a [`Tape`], exactly like the
//! llama example: the trace is frozen per sequence length and replayed on
//! every later step. Weight tensors stay in their gguf PQ2_0 super-block form;
//! [`QWeight::dequantize`] is part of the traced graph, so the egraph decides
//! whether to fuse the dequant into the projection or materialize it.
//!
//! Every dtype choice follows llama.cpp: activations, norms, conv, delta-net
//! state, logits and the rotation are F32; the full-attention K/V cache is
//! F16 (chosen by the CLI `--max-context` owner).

use std::collections::HashMap;
use std::path::Path;

use zyx::{DType, Dev, FrozenTape, QDType, Tape, Tensor, ZyxError};
use zyx_nn::RMSNorm;

use crate::config::Config;
use crate::prism::Prism;
use crate::weights::QWeight;

/// Elements covered by one PQ2_0 super-block.
const PQ2_BLOCK: i64 = 128;
/// Gated-delta-rule chunk length used by llama.cpp's chunked prefill.
const CHUNK: i64 = 64;

/// SwiGLU weights.
struct Ffn {
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    norm: Tensor,
}

/// One full-attention block and its F16 key/value cache.
struct FullBlock {
    norm: Tensor,
    q: QWeight,
    k: QWeight,
    v: QWeight,
    o: QWeight,
    q_norm: Tensor,
    k_norm: Tensor,
    ffn: Ffn,
    cache_k: Tensor,
    cache_v: Tensor,
}

/// One gated-delta-net block plus its conv history and recurrent state.
struct LinearBlock {
    norm: Tensor,
    qkv: QWeight,
    gate: QWeight,
    beta: Tensor,
    alpha: Tensor,
    conv_w: Tensor,
    ssm_a: Tensor,
    ssm_dt: Tensor,
    ssm_norm: Tensor,
    ssm_out: QWeight,
    ffn: Ffn,
    conv_state: Tensor,
    state: Tensor,
}

enum Block {
    Full(FullBlock),
    Linear(LinearBlock),
}

/// RoPE tables for every position the cache can hold.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

/// A frozen step. The key is `(sequence length, mask start)`; decode steps use
/// `(1, 0)` because a single query needs no causal mask and the position is a
/// replay variable.
struct Plan {
    frozen: FrozenTape,
}

/// Ternary-Bonsai-2-27B.
pub struct Bonsai {
    cfg: Config,
    prism: Prism,
    emb: QWeight,
    output: QWeight,
    output_norm: Tensor,
    blocks: Vec<Block>,
    rope: Rope,
    max_context: i64,
    /// Padded prefill width; a multiple of the GDN chunk length.
    max_prefill: i64,
    plans: HashMap<i64, Plan>,
}

impl Bonsai {
    /// Loads the trunk from a GGUF file.
    ///
    /// `max_context` sizes the full-attention K/V caches. `max_prefill` is the
    /// fixed padded width every prompt chunk is traced at; it is rounded up to
    /// a multiple of the 64-token GDN chunk. Decode always uses one token.
    ///
    /// # Errors
    /// Missing/mismatched metadata, unsupported tensor formats, or device
    /// allocation failures.
    pub fn from_gguf(path: impl AsRef<Path>, max_context: i64, max_prefill: i64) -> Result<Self, ZyxError> {
        let max_prefill = (max_prefill + CHUNK - 1) / CHUNK * CHUNK;
        if max_prefill <= 0 {
            return Err(ZyxError::shape_error("max_prefill must be positive".into()));
        }
        let (metadata, mut host) = Tensor::load_gguf(path)?;
        let cfg = Config::from_metadata(&metadata)?;
        let mut prism = Prism::from_metadata(&metadata)?;
        let dev = compute_device();
        prism.to(dev)?;

        let mut take = |name: &str| -> Result<Tensor, ZyxError> {
            host.remove(name)
                .ok_or_else(|| ZyxError::parse_error(format!("gguf is missing tensor `{name}`").into()))?
                .to(dev)
        };
        let emb_blocks = take("token_embd.weight")?;
        let vocab = emb_blocks.resolve_shape()[0] * PQ2_BLOCK / cfg.hidden;
        let emb = QWeight::new(emb_blocks, vocab, cfg.hidden)?;
        let mut output = QWeight::new(take("output.weight")?, vocab, cfg.hidden)?;
        output.folded = prism.is_folded("output.weight");
        let output_norm = take("output_norm.weight")?;

        let mut blocks = Vec::with_capacity(usize::try_from(cfg.n_blocks).expect("block count fits usize"));
        for i in 0..cfg.n_blocks {
            let pre = format!("blk.{i}.");
            let ffn = Ffn {
                gate: quant(&mut take, &prism, &(pre.clone() + "ffn_gate.weight"), cfg.ffn, cfg.hidden)?,
                up: quant(&mut take, &prism, &(pre.clone() + "ffn_up.weight"), cfg.ffn, cfg.hidden)?,
                down: quant(&mut take, &prism, &(pre.clone() + "ffn_down.weight"), cfg.hidden, cfg.ffn)?,
                norm: take(&(pre.clone() + "post_attention_norm.weight"))?,
            };
            let norm = take(&(pre.clone() + "attn_norm.weight"))?;
            if cfg.is_full(i) {
                let q_dim = 2 * cfg.n_head * cfg.head_dim;
                let kv_dim = cfg.n_kv * cfg.head_dim;
                blocks.push(Block::Full(FullBlock {
                    norm,
                    q: quant(&mut take, &prism, &(pre.clone() + "attn_q.weight"), q_dim, cfg.hidden)?,
                    k: quant(&mut take, &prism, &(pre.clone() + "attn_k.weight"), kv_dim, cfg.hidden)?,
                    v: quant(&mut take, &prism, &(pre.clone() + "attn_v.weight"), kv_dim, cfg.hidden)?,
                    o: quant(&mut take, &prism, &(pre.clone() + "attn_output.weight"), cfg.hidden, cfg.n_head * cfg.head_dim)?,
                    q_norm: take(&(pre.clone() + "attn_q_norm.weight"))?,
                    k_norm: take(&(pre.clone() + "attn_k_norm.weight"))?,
                    ffn,
                    cache_k: Tensor::zeros([max_context, cfg.n_kv, cfg.head_dim], DType::F16).contiguous()?,
                    cache_v: Tensor::zeros([max_context, cfg.n_kv, cfg.head_dim], DType::F16).contiguous()?,
                }));
            } else {
                blocks.push(Block::Linear(LinearBlock {
                    norm,
                    qkv: quant(&mut take, &prism, &(pre.clone() + "attn_qkv.weight"), cfg.conv_dim(), cfg.hidden)?,
                    gate: quant(&mut take, &prism, &(pre.clone() + "attn_gate.weight"), cfg.ssm_value_dim(), cfg.hidden)?,
                    beta: take(&(pre.clone() + "ssm_beta.weight"))?.cast(DType::F32),
                    alpha: take(&(pre.clone() + "ssm_alpha.weight"))?.cast(DType::F32),
                    conv_w: take(&(pre.clone() + "ssm_conv1d.weight"))?,
                    ssm_a: take(&(pre.clone() + "ssm_a"))?,
                    ssm_dt: take(&(pre.clone() + "ssm_dt.bias"))?,
                    ssm_norm: take(&(pre.clone() + "ssm_norm.weight"))?,
                    ssm_out: quant(
                        &mut take,
                        &prism,
                        &(pre.clone() + "ssm_out.weight"),
                        cfg.hidden,
                        cfg.ssm_value_dim(),
                    )?,
                    ffn,
                    conv_state: Tensor::zeros([cfg.conv_kernel - 1, cfg.conv_dim()], DType::F32).contiguous()?,
                    state: Tensor::zeros([cfg.ssm_v_heads, cfg.ssm_state, cfg.ssm_state], DType::F32).contiguous()?,
                }));
            }
        }

        let rope = Rope::new(&cfg, max_context + max_prefill)?.to(dev)?;
        Ok(Self {
            cfg,
            prism,
            emb,
            output,
            output_norm,
            blocks,
            rope,
            max_context,
            max_prefill,
            plans: HashMap::new(),
        })
    }

    /// Runs one step and returns logits `[1, vocab]` for the last real token.
    ///
    /// Prompt chunks are padded to `max_prefill`; decode is the one-token
    /// plan. Both plans keep the real length and start position as runtime
    /// variables, so one frozen graph serves every prompt and every decode
    /// position.
    ///
    /// # Errors
    /// Graph construction, freeze, replay, or an oversized prompt chunk.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: i64) -> Result<Tensor, ZyxError> {
        let real = input_ids.numel().item::<i64>();
        if start_pos + real > self.max_context {
            return Err(ZyxError::shape_error(
                format!("step needs positions {}..{}, above max_context {}", start_pos, start_pos + real - 1, self.max_context).into(),
            ));
        }
        let (padded, ids_host, valid_host) = if real == 1 {
            let id = input_ids.cast(DType::I64).reshape([1i64])?.to_vec::<i64>()?[0];
            (1, vec![id], vec![1.0f32])
        } else {
            if real > self.max_prefill {
                return Err(ZyxError::shape_error(
                    format!("prompt chunk has {real} tokens, above max_prefill {}", self.max_prefill).into(),
                ));
            }
            let ids = input_ids.cast(DType::I64).reshape([real])?.to_vec::<i64>()?;
            let mut padded_ids = vec![0i64; usize::try_from(self.max_prefill).expect("max_prefill fits usize")];
            padded_ids[..usize::try_from(real).expect("real fits usize")].copy_from_slice(&ids);
            let mut valid = vec![0.0f32; usize::try_from(self.max_prefill).expect("max_prefill fits usize")];
            valid[..usize::try_from(real).expect("real fits usize")].fill(1.0);
            (self.max_prefill, padded_ids, valid)
        };
        let ids = Tensor::from_vec(ids_host, [padded])?;
        let valid = Tensor::from_vec(valid_host, [padded])?;
        let pos = Tensor::variable(start_pos);
        let real_t = Tensor::variable(real);
        let cache_len = &pos + &real_t;
        if let Some(plan) = self.plans.get(&padded) {
            return Ok(plan.frozen.replay([&ids, &valid, &pos, &real_t])?.remove(0));
        }

        let tape = Tape::new(self.params())?;
        tape.add(&ids)?;
        tape.add(&valid)?;

        let Self {
            cfg,
            prism,
            emb,
            output,
            output_norm,
            blocks,
            rope,
            max_context,
            ..
        } = self;

        // Causal mask built from runtime position variables: rows are the
        // padded queries, columns are the cache positions actually read.
        let mask = if padded == 1 {
            Tensor::zeros([1i64, 1i64], DType::F32)
        } else {
            let rows = Tensor::arange(0i64, padded, 1i64)?.reshape([padded, 1i64])?;
            let cols = Tensor::arange(0i64, *max_context, 1i64)?.reshape([1i64, *max_context])?;
            let limit = &pos + &rows + 1i64;
            let keep = cols.cmplt(limit)?;
            let mask = (keep.cast(DType::F32) - 1.0f32) * 1e30f32;
            mask.narrow(1, 0i64, cache_len.clone())?
        };

        // Embedding: row-gather the folded PQ2_0 table, then restore the
        // primal basis with `s * (H z)`. Padded rows are zeroed so they can
        // never leak through the conv or the state.
        let blocks_per_row = cfg.hidden / PQ2_BLOCK;
        let row = ids.reshape([padded, 1i64])? * blocks_per_row;
        let col = Tensor::arange(0i64, blocks_per_row, 1i64)?.reshape([1i64, blocks_per_row])?;
        let index = (row + col).reshape([padded * blocks_per_row])?.cast(DType::I32);
        let picked = emb.blocks.index_select(0, index)?;
        let mut x = picked.dequantize(QDType::PQ2_0, DType::F32)?.reshape([padded, cfg.hidden])?;
        if prism.is_inverse("token_embd.weight") {
            x = prism.inverse(&x, cfg.hidden)?;
        }
        x = x * valid.unsqueeze(1)?;

        for block in blocks.iter_mut() {
            match block {
                Block::Full(b) => {
                    let h = b.forward(cfg, prism, &rope, &mask, &x, &pos, &cache_len, padded)?;
                    x = ffn_forward(prism, &b.ffn, &h, cfg.eps)?;
                }
                Block::Linear(b) => {
                    let h = b.forward(cfg, prism, &x, &pos, &real_t, &valid, padded)?;
                    x = ffn_forward(prism, &b.ffn, &h, cfg.eps)?;
                }
            }
        }

        // Only the last real token needs logits.
        let last = x.narrow(0, &real_t - 1i64, 1i64)?;
        let last = RMSNorm {
            weight: output_norm.clone(),
            eps: cfg.eps,
        }
        .forward(&last)?;
        let logits = linear(prism, &last, output)?;

        let mut outputs: Vec<Tensor> = vec![logits.clone()];
        for block in blocks.iter() {
            match block {
                Block::Full(b) => {
                    outputs.push(b.cache_k.clone());
                    outputs.push(b.cache_v.clone());
                }
                Block::Linear(b) => {
                    outputs.push(b.conv_state.clone());
                    outputs.push(b.state.clone());
                }
            }
        }
        let frozen = tape.freeze([&ids, &valid, &pos, &real_t], &outputs)?;
        let res = frozen.replay([&ids, &valid, &pos, &real_t])?;
        self.plans.insert(padded, Plan { frozen });
        Ok(res.into_iter().next().expect("logits is the first output"))
    }

    /// Every tensor the tape promotes: weights and owned caches.
    fn params(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.emb.blocks, &self.output.blocks, &self.output_norm];
        for block in &self.blocks {
            match block {
                Block::Full(b) => {
                    params.extend([&b.norm, &b.q_norm, &b.k_norm, &b.cache_k, &b.cache_v]);
                    params.extend([&b.q.blocks, &b.k.blocks, &b.v.blocks, &b.o.blocks]);
                    params.extend([&b.ffn.norm, &b.ffn.gate.blocks, &b.ffn.up.blocks, &b.ffn.down.blocks]);
                }
                Block::Linear(b) => {
                    params.extend([
                        &b.norm,
                        &b.beta,
                        &b.alpha,
                        &b.conv_w,
                        &b.ssm_a,
                        &b.ssm_dt,
                        &b.ssm_norm,
                        &b.conv_state,
                        &b.state,
                    ]);
                    params.extend([&b.qkv.blocks, &b.gate.blocks, &b.ssm_out.blocks]);
                    params.extend([&b.ffn.norm, &b.ffn.gate.blocks, &b.ffn.up.blocks, &b.ffn.down.blocks]);
                }
            }
        }
        params
    }
}

impl Rope {
    /// Builds the F32 cos/sin tables for llama.cpp's interleaved MRoPE with
    /// text-only positions (all axes share `pos`).
    fn new(cfg: &Config, max_context: i64) -> Result<Self, ZyxError> {
        let half = cfg.rot_dim / 2;
        let mut cos = vec![0.0f32; (max_context * half) as usize];
        let mut sin = vec![0.0f32; (max_context * half) as usize];
        for pos in 0..max_context {
            for pair in 0..half {
                let angle = pos as f64 * cfg.rope_theta.powf(-2.0 * pair as f64 / cfg.rot_dim as f64);
                let (s, c) = angle.sin_cos();
                cos[(pos * half + pair) as usize] = c as f32;
                sin[(pos * half + pair) as usize] = s as f32;
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, [max_context, half])?,
            sin: Tensor::from_vec(sin, [max_context, half])?,
        })
    }

    /// Moves both tables to `dev`.
    fn to(self, dev: Dev) -> Result<Self, ZyxError> {
        Ok(Self {
            cos: self.cos.to(dev)?,
            sin: self.sin.to(dev)?,
        })
    }

    /// Partial RoPE on `[seq, heads, head_dim]`. llama.cpp pairs dimension
    /// `j` with `j + rot_dim/2`; the untouched tail is concatenated back.
    fn apply(&self, x: &Tensor, pos: &Tensor, seq: i64, rot_dim: i64, head_dim: i64) -> Result<Tensor, ZyxError> {
        let cos = self.cos.narrow(0, pos.clone(), seq)?;
        let sin = self.sin.narrow(0, pos.clone(), seq)?;
        let x = x.transpose(0, 1)?.unsqueeze(0)?;
        let rotated = x.narrow(3, 0i64, rot_dim)?;
        let rest = x.narrow(3, rot_dim, head_dim - rot_dim)?;
        let rotated = rotated.rope(&cos, &sin)?;
        Tensor::cat([&rotated, &rest], 3)?.squeeze([0]).transpose(0, 1)
    }
}

impl FullBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        cfg: &Config,
        prism: &Prism,
        rope: &Rope,
        mask: &Tensor,
        x: &Tensor,
        pos: &Tensor,
        cache_len: &Tensor,
        seq: i64,
    ) -> Result<Tensor, ZyxError> {
        let h = RMSNorm {
            weight: self.norm.clone(),
            eps: cfg.eps,
        }
        .forward(x)?;
        let qg = linear(prism, &h, &self.q)?;
        let qg = qg.reshape([seq, cfg.n_head, 2i64, cfg.head_dim])?;
        let q = qg.narrow(2, 0i64, 1i64)?.squeeze([2]);
        let gate = qg.narrow(2, 1i64, 1i64)?.squeeze([2]);
        let q = RMSNorm {
            weight: self.q_norm.clone(),
            eps: cfg.eps,
        }
        .forward(&q)?;
        let k = linear(prism, &h, &self.k)?.reshape([seq, cfg.n_kv, cfg.head_dim])?;
        let k = RMSNorm {
            weight: self.k_norm.clone(),
            eps: cfg.eps,
        }
        .forward(&k)?;
        let v = linear(prism, &h, &self.v)?.reshape([seq, cfg.n_kv, cfg.head_dim])?;

        let q = rope.apply(&q, pos, seq, cfg.rot_dim, cfg.head_dim)?;
        let k = rope.apply(&k, pos, seq, cfg.rot_dim, cfg.head_dim)?;

        let real = cache_len - pos;
        self.cache_k
            .clone()
            .narrow(0, pos.clone(), real.clone())?
            .assign(&k.narrow(0, 0i64, real.clone())?.cast(DType::F16))?;
        self.cache_v
            .clone()
            .narrow(0, pos.clone(), real.clone())?
            .assign(&v.narrow(0, 0i64, real)?.cast(DType::F16))?;

        let kk = self.cache_k.narrow(0, 0i64, cache_len.clone())?.transpose(0, 1)?;
        let vv = self.cache_v.narrow(0, 0i64, cache_len.clone())?.transpose(0, 1)?;
        let rep = cfg.n_head / cfg.n_kv;
        let kk = kk
            .unsqueeze(1)?
            .expand([-1i64, rep, -1, -1])?
            .reshape([Tensor::from(cfg.n_head), cache_len.clone(), Tensor::from(cfg.head_dim)])?;
        let vv = vv
            .unsqueeze(1)?
            .expand([-1i64, rep, -1, -1])?
            .reshape([Tensor::from(cfg.n_head), cache_len.clone(), Tensor::from(cfg.head_dim)])?;

        let q = q.transpose(0, 1)?;
        let scale = Tensor::from(1.0f32 / (cfg.head_dim as f32).sqrt());
        let attn = q
            .cast(DType::F16)
            .dot_dtype_output(kk.transpose(1, 2)?.cast(DType::F16), DType::F32, DType::F32)?
            * scale;
        let attn = if seq == 1 {
            attn
        } else {
            attn + mask.unsqueeze(0)?
        };
        let probs = attn.softmax([-1])?;
        let out = probs
            .cast(DType::F16)
            .dot_dtype_output(&vv, DType::F32, DType::F32)?
            .transpose(0, 1)?
            .reshape([seq, cfg.n_head * cfg.head_dim])?;
        let gate = gate.reshape([seq, cfg.n_head * cfg.head_dim])?;
        let out = out * gate.sigmoid();
        let o = linear(prism, &out, &self.o)?;
        Ok(x + &o)
    }
}

impl LinearBlock {
    fn forward(
        &mut self,
        cfg: &Config,
        prism: &Prism,
        x: &Tensor,
        pos: &Tensor,
        real: &Tensor,
        valid: &Tensor,
        seq: i64,
    ) -> Result<Tensor, ZyxError> {
        let h = RMSNorm {
            weight: self.norm.clone(),
            eps: cfg.eps,
        }
        .forward(x)?;
        // A fresh sequence (position 0) starts from the zero recurrent state
        // and zero conv history, like llama.cpp's reset. Multiplying by
        // `1 - (pos == 0)` keeps the reset a runtime scalar, so one frozen
        // plan serves both a fresh prompt and a continuing decode.
        let keep = Tensor::from(1.0f32) - pos.equal(0i64)?.cast(DType::F32);
        let history = &self.conv_state * &keep;
        let state = &self.state * &keep;

        let qkv = linear(prism, &h, &self.qkv)?;
        let z = linear(prism, &h, &self.gate)?;
        let beta = linear_input(&h, &self.beta)?;
        let alpha = linear_input(&h, &self.alpha)?;

        // Depthwise causal conv over history + this chunk, then SiLU.
        let cat = Tensor::cat([&history, &qkv], 0)?;
        let mut mixed = Tensor::zeros([seq, cfg.conv_dim()], DType::F32);
        for k in 0..cfg.conv_kernel {
            let window = cat.narrow(0, k, seq)?;
            let weight = self.conv_w.narrow(1, k, 1i64)?.squeeze([1]);
            mixed = mixed + window * weight.unsqueeze(0)?;
        }
        let mixed = mixed.swish();
        // Keep the last three *real* rows; the padded tail is not history.
        let conv_state = cat.narrow(0, real.clone(), cfg.conv_kernel - 1)?.contiguous()?;
        self.conv_state.clone().assign(&conv_state)?;

        let (q, k, v) = split_qkv(cfg, &mixed, seq)?;
        let eps = cfg.eps;
        let q = &q * ((q.square().sum_keepdim([-1])? + eps).rsqrt());
        let k = &k * ((k.square().sum_keepdim([-1])? + eps).rsqrt());
        let scale = Tensor::from(1.0f32 / (cfg.ssm_state as f32).sqrt());
        let q = q * scale;

        let rep = cfg.ssm_v_heads / cfg.ssm_k_heads;
        let q = repeat_tiled(&q, rep)?;
        let k = repeat_tiled(&k, rep)?;

        let b = beta.sigmoid() * valid.unsqueeze(1)?;
        let g = ((&alpha + &self.ssm_dt).softplus(1.0f32, 20.0f32) * &self.ssm_a) * valid.unsqueeze(1)?;
        let (core, new_state) = if seq == 1 {
            delta_step(cfg, &q, &k, &v, &g, &b, &state)?
        } else {
            delta_chunked(cfg, &q, &k, &v, &g, &b, &state)?
        };
        self.state.clone().assign(&new_state)?;

        let core = core.reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])?;
        let z = z.reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])?;
        let normed = RMSNorm {
            weight: self.ssm_norm.clone(),
            eps: cfg.eps,
        }
        .forward(&core)?;
        let normed = normed * z.swish();
        let mut normed = normed.reshape([seq, cfg.ssm_value_dim()])?;

        // The folded out-projection is stored in grouped V order; the
        // activation leaves `delta_*` in llama.cpp's tiled order.
        if prism.gdn_v_grouped {
            let vd = cfg.ssm_v_head_dim;
            let nk = cfg.ssm_k_heads;
            let rep = cfg.ssm_v_heads / nk;
            normed = normed
                .reshape([seq, rep, nk, vd])?
                .transpose(1, 2)?
                .reshape([seq, cfg.ssm_value_dim()])?;
        }

        let o = linear(prism, &normed, &self.ssm_out)?;
        Ok(x + &o)
    }
}

/// A quantized projection: F16 operands on the tensor cores, F32 accumulator
/// and F32 result (llama.cpp's `ggml_mul_mat` contract, tinygrad's WMMA path).
fn linear(prism: &Prism, x: &Tensor, w: &QWeight) -> Result<Tensor, ZyxError> {
    let x = if w.folded { prism.forward(x, w.cols)? } else { x.clone() };
    x.cast(DType::F16)
        .dot_dtype_output(w.dequantize(DType::F16)?.t(), DType::F32, DType::F32)
}

/// A dense BF16/F32 projection (the small `ssm_beta`/`ssm_alpha` weights),
/// with the same F16-operand/F32-accumulator contract.
fn linear_input(x: &Tensor, weight: &Tensor) -> Result<Tensor, ZyxError> {
    x.cast(DType::F16)
        .dot_dtype_output(weight.cast(DType::F16).t(), DType::F32, DType::F32)
}

/// SwiGLU feed-forward with its own pre-norm and residual.
fn ffn_forward(prism: &Prism, ffn: &Ffn, x: &Tensor, eps: f64) -> Result<Tensor, ZyxError> {
    let h = RMSNorm {
        weight: ffn.norm.clone(),
        eps,
    }
    .forward(x)?;
    let gate = linear(prism, &h, &ffn.gate)?;
    let up = linear(prism, &h, &ffn.up)?;
    let mid = gate.swish() * up;
    let down = linear(prism, &mid, &ffn.down)?;
    Ok(x + &down)
}

/// Splits the mixed conv output into q/k/v head tensors.
fn split_qkv(cfg: &Config, mixed: &Tensor, seq: i64) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
    let kd = cfg.ssm_key_dim();
    let q = mixed.narrow(1, 0i64, kd)?.reshape([seq, cfg.ssm_k_heads, cfg.ssm_state])?;
    let k = mixed.narrow(1, kd, kd)?.reshape([seq, cfg.ssm_k_heads, cfg.ssm_state])?;
    let v = mixed
        .narrow(1, 2 * kd, cfg.ssm_value_dim())?
        .reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])?;
    Ok((q, k, v))
}

/// Repeats `H_k` key heads to `H_v` value heads in llama.cpp's tiled order
/// (`v_head % H_k`), not the interleaved HF order.
fn repeat_tiled(x: &Tensor, rep: i64) -> Result<Tensor, ZyxError> {
    let [seq, heads, dim] = x.dims::<3>()?;
    x.unsqueeze(1)?
        .expand([-1i64, rep, -1, -1])?
        .reshape([seq, heads * rep, dim])
}

/// Llama.cpp's token-recurrent gated delta rule (one decode step).
fn delta_step(
    cfg: &Config,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    b: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor), ZyxError> {
    let q = q.squeeze([0]);
    let k = k.squeeze([0]);
    let v = v.squeeze([0]);
    let g = g.squeeze([0]).exp();
    let b = b.squeeze([0]);

    let kv = (state * k.unsqueeze(2)?).sum([1])?;
    let delta = (v - g.unsqueeze(1)? * kv) * b.unsqueeze(1)?;
    let state = state * g.unsqueeze(1)?.unsqueeze(1)? + k.unsqueeze(2)? * delta.unsqueeze(1)?;
    let core = (state.clone() * q.unsqueeze(2)?).sum([1])?;
    let _ = cfg;
    Ok((core.unsqueeze(0)?, state))
}

/// Llama.cpp's chunked gated delta rule (prefill).
fn delta_chunked(
    cfg: &Config,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    b: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor), ZyxError> {
    let [seq, heads, dim] = q.dims::<3>()?;
    let seq = seq.item::<i64>();
    let h = heads.item::<i64>();
    let d = dim.item::<i64>();
    let pad = (CHUNK - seq % CHUNK) % CHUNK;
    let chunks = (seq + pad) / CHUNK;

    let pad_t = |x: &Tensor| -> Result<Tensor, ZyxError> {
        if pad == 0 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        let mut pad_shape = shape.clone();
        pad_shape[1] = Tensor::from(pad);
        let zeros = Tensor::zeros(pad_shape, DType::F32);
        Tensor::cat([x, &zeros], 1)
    };

    let q = q.transpose(0, 1)?;
    let k = k.transpose(0, 1)?;
    let v = v.transpose(0, 1)?;
    let g = g.transpose(0, 1)?;
    let b = b.transpose(0, 1)?;
    let q = pad_t(&q)?;
    let k = pad_t(&k).and_then(|x| x.reshape([h, chunks, CHUNK, d]))?;
    let v = pad_t(&v).and_then(|x| x.reshape([h, chunks, CHUNK, d]))?;
    let g = pad_t(&g).and_then(|x| x.reshape([h, chunks, CHUNK]))?;
    let b = pad_t(&b).and_then(|x| x.reshape([h, chunks, CHUNK]))?;
    let q = q.reshape([h, chunks, CHUNK, d])?;

    let gcs = g.cumsum(2)?;
    let gexp = gcs.exp();
    let gi = gcs.unsqueeze(3)?;
    let gj = gcs.unsqueeze(2)?;
    let decay = (gi - gj).exp() * lower(CHUNK, 0)?;
    let strict = lower(CHUNK, -1)?;

    let b3 = b.unsqueeze(3)?;
    let v_b = &v * &b3;
    let k_b = &k * &b3;

    let a = k_b.matmul(&k.transpose(2, 3)?)? * &decay * &strict;
    let n = -a;
    let identity = Tensor::eye(CHUNK, DType::F32)
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand([h, chunks, CHUNK, CHUNK])?;
    let mut m = identity;
    let mut npow = n.clone();
    for _ in 0..6 {
        let next = &m + npow.matmul(&m)?;
        m = next;
        npow = npow.matmul(&npow)?;
    }

    let v_new = m.matmul(&v_b)?;
    let k_cd = m.matmul(&(k_b * gexp.unsqueeze(3)?))?;
    let q_g = &q * gexp.unsqueeze(3)?;

    let mut state = state.clone();
    let mut cores = Vec::with_capacity(usize::try_from(chunks).expect("chunk count fits usize"));
    for chunk in 0..chunks {
        let qc = q_g.narrow(1, chunk, 1i64)?.squeeze([1]);
        let kc = k.narrow(1, chunk, 1i64)?.squeeze([1]);
        let kcd = k_cd.narrow(1, chunk, 1i64)?.squeeze([1]);
        let vnc = v_new.narrow(1, chunk, 1i64)?.squeeze([1]);
        let gc = gcs.narrow(1, chunk, 1i64)?.squeeze([1]);
        let dc = decay.narrow(1, chunk, 1i64)?.squeeze([1]);

        let a2 = qc.matmul(&kc.transpose(1, 2)?)? * &dc * &strict;
        let v_prime = kcd.matmul(&state)?;
        let v_new_c = vnc - v_prime;
        let core = qc.matmul(&state)? + a2.matmul(&v_new_c)?;

        let g_last = gc.narrow(1, CHUNK - 1, 1i64)?;
        let kg = &kc * (&g_last - &gc).exp().unsqueeze(2)?;
        state = state * g_last.exp().unsqueeze(2)? + kg.transpose(1, 2)?.matmul(&v_new_c)?;
        cores.push(core);
    }

    let core = Tensor::cat(&cores, 1)?;
    let core = core.transpose(0, 1)?.narrow(0, 0i64, seq)?;
    let _ = cfg;
    Ok((core, state))
}

/// A `[n, n]` F32 lower-triangular mask (`diagonal` = 0 keeps the diagonal).
fn lower(n: i64, diagonal: i64) -> Result<Tensor, ZyxError> {
    let mut data = vec![0.0f32; (n * n) as usize];
    for i in 0..n {
        for j in 0..n {
            if j <= i + diagonal {
                data[(i * n + j) as usize] = 1.0;
            }
        }
    }
    Tensor::from_vec(data, [n, n])
}

/// Loads one quantized projection and records whether it is folded.
fn quant(
    take: &mut impl FnMut(&str) -> Result<Tensor, ZyxError>,
    prism: &Prism,
    name: &str,
    rows: i64,
    cols: i64,
) -> Result<QWeight, ZyxError> {
    let mut weight = QWeight::new(take(name)?, rows, cols)?;
    weight.folded = prism.is_folded(name);
    Ok(weight)
}

/// The device weights live on. `ZYX_BONSAI_DEVICE` overrides the pick.
fn compute_device() -> Dev {
    if let Ok(s) = std::env::var("ZYX_BONSAI_DEVICE") {
        return s.parse().unwrap_or_else(|e| panic!("ZYX_BONSAI_DEVICE: {e}"));
    }
    Dev::all().into_iter().find(|d| !matches!(d, Dev::C)).unwrap_or(Dev::C)
}
