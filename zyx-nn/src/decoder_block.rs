// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Pre-norm decoder block: attention and feed-forward each wrapped in a
//! residual with an RMSNorm on the branch input.
//!
//! The block owns the projections directly and names them after the GGUF
//! tensors (`blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`, ...), so a
//! checkpoint binds through `Module::set_params` with no renaming layer.
//! GGUF has no level between `blk.N.` and `attn_q`, which is why the
//! attention and the SwiGLU feed-forward are fields here rather than
//! submodules.

use crate::gqa_attention::{repeat_kv, rotate, RopeScaling};
use crate::{Linear, RMSNorm};
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Pre-norm decoder block with grouped-query attention and a SwiGLU MLP.
#[derive(Debug, Clone, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct DecoderBlock {
    /// Query projection
    pub attn_q: Linear,
    /// Key projection
    pub attn_k: Linear,
    /// Value projection
    pub attn_v: Linear,
    /// Attention output projection
    pub attn_output: Linear,
    /// Feed-forward gate projection
    pub ffn_gate: Linear,
    /// Feed-forward up projection
    pub ffn_up: Linear,
    /// Feed-forward down projection
    pub ffn_down: Linear,
    /// pre-attention norm
    pub attn_norm: RMSNorm,
    /// pre-MLP norm
    pub ffn_norm: RMSNorm,
    /// Key cache, `[max_context, n_kv_head, head_dim]`. Owned state: written
    /// in place by every forward pass, so it must be realized alongside the
    /// outputs.
    pub cache_k: Tensor,
    /// Value cache, same shape as [`Self::cache_k`].
    pub cache_v: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
}

impl DecoderBlock {
    /// Builds the block with random projections and a zeroed cache.
    ///
    /// `head_dim`, `max_position_embeddings`, `theta` and `scaling` parametrise
    /// the rotary frequency table that [`Self::new`] builds and stores as
    /// `cos`/`sin`. `scaling` is llama-3's piecewise rescale of the
    /// low-frequency band, or `None` for plain RoPE.
    ///
    /// # Errors
    ///
    /// Returns a shape error if any head/dimension combination is invalid.
    pub fn new(
        n_embd: i64,
        intermediate: i64,
        n_head: i64,
        n_kv_head: i64,
        head_dim: i64,
        max_position_embeddings: i64,
        theta: f64,
        scaling: Option<RopeScaling>,
        max_context: i64,
        eps: f64,
        dtype: DType,
    ) -> Result<DecoderBlock, ZyxError> {
        if n_head % n_kv_head != 0 {
            return Err(ZyxError::shape_error(format!("n_head {n_head} is not a multiple of n_kv_head {n_kv_head}").into()));
        }
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;
        // Rotary frequency table, `[max_position_embeddings, head_dim / 2]`:
        // `cos`/`sin` of `position * inv_freq`, with llama-3's piecewise
        // rescale of the low-frequency band when `scaling` is set.
        let mut inv_freq: Vec<f32> =
            (0..head_dim).step_by(2).map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32).collect();
        if let Some(scaling) = &scaling {
            let low_wavelen = scaling.original_max_position_embeddings as f64 / scaling.low_freq_factor;
            let high_wavelen = scaling.original_max_position_embeddings as f64 / scaling.high_freq_factor;
            for freq in inv_freq.iter_mut() {
                let wavelen = 2.0 * std::f64::consts::PI / *freq as f64;
                if wavelen > low_wavelen {
                    *freq = (*freq as f64 / scaling.factor) as f32;
                } else if wavelen < high_wavelen {
                    // Unchanged: high frequencies keep the original spacing.
                } else {
                    let smooth = (scaling.original_max_position_embeddings as f64 / wavelen - scaling.low_freq_factor)
                        / (scaling.high_freq_factor - scaling.low_freq_factor);
                    *freq = ((1.0 - smooth) * *freq as f64 / scaling.factor + smooth * *freq as f64) as f32;
                }
            }
        }
        let inv_len = inv_freq.len() as i64;
        let inv_freq = Tensor::from(inv_freq).reshape([1, inv_len]).unwrap();
        let t = Tensor::arange(0u32, max_position_embeddings as u32, 1)
            .unwrap()
            .cast(DType::F32)
            .reshape([max_position_embeddings, 1])
            .unwrap();
        let freqs = t.matmul(&inv_freq).unwrap();
        let cos = freqs.cos().cast(dtype);
        let sin = freqs.sin().cast(dtype);
        let mut attn_norm = RMSNorm::new(n_embd, dtype);
        attn_norm.eps = eps;
        let mut ffn_norm = RMSNorm::new(n_embd, dtype);
        ffn_norm.eps = eps;
        Ok(DecoderBlock {
            attn_q: Linear::new(n_embd, q_dim, false, dtype)?,
            attn_k: Linear::new(n_embd, kv_dim, false, dtype)?,
            attn_v: Linear::new(n_embd, kv_dim, false, dtype)?,
            attn_output: Linear::new(q_dim, n_embd, false, dtype)?,
            ffn_gate: Linear::new(n_embd, intermediate, false, dtype)?,
            ffn_up: Linear::new(n_embd, intermediate, false, dtype)?,
            ffn_down: Linear::new(intermediate, n_embd, false, dtype)?,
            attn_norm,
            ffn_norm,
            cache_k: Tensor::zeros([max_context, n_kv_head, head_dim], dtype).contiguous()?,
            cache_v: Tensor::zeros([max_context, n_kv_head, head_dim], dtype).contiguous()?,
            cos,
            sin,
            num_heads: n_head,
            n_kv_heads: n_kv_head,
            head_dim,
        })
    }

    /// Forward pass.
    ///
    /// `xs` is `[batch, seq_len, n_embd]`. `start_pos` is the absolute cache
    /// position of the first token; `cache_len` is how many cached positions
    /// the attention reads (`start_pos + seq_len`). Both are symbolic tensors.
    ///
    /// # Errors
    ///
    /// Returns a shape error if `xs` is not rank 3 or the cache is too short.
    pub fn forward(&mut self, xs: &Tensor, start_pos: &Tensor, cache_len: &Tensor) -> Result<Tensor, ZyxError> {
        let residual = xs.clone();
        let xs = self.attn_norm.forward(xs)?;
        let [b_size, seq_len, _n_embd] = xs.dims::<3>()?;
        let q = self.attn_q.forward(&xs)?;
        let k = self.attn_k.forward(&xs)?;
        let v = self.attn_v.forward(&xs)?;

        let q = q.reshape([b_size.clone(), seq_len.clone(), self.num_heads.into(), self.head_dim.into()])?.transpose(1, 2)?;
        let k = k.reshape([b_size.clone(), seq_len.clone(), self.n_kv_heads.into(), self.head_dim.into()])?.transpose(1, 2)?;
        let v = v.reshape([b_size.clone(), seq_len.clone(), self.n_kv_heads.into(), self.head_dim.into()])?.transpose(1, 2)?;

        let q = rotate(&q, &self.cos, &self.sin, start_pos)?;
        let k = rotate(&k, &self.cos, &self.sin, start_pos)?;

        let k_new = k.squeeze([0]).transpose(0, 1)?.cast(self.cache_k.dtype());
        let v_new = v.squeeze([0]).transpose(0, 1)?.cast(self.cache_v.dtype());
        self.cache_k.narrow(0, start_pos, &seq_len)?.assign(&k_new)?;
        self.cache_v.narrow(0, start_pos, &seq_len)?.assign(&v_new)?;

        let k = self.cache_k.narrow(0, 0i64, cache_len.clone())?.unsqueeze(0)?.transpose(1, 2)?;
        let v = self.cache_v.narrow(0, 0i64, cache_len.clone())?.unsqueeze(0)?.transpose(1, 2)?;

        let n_rep = (self.num_heads / self.n_kv_heads) as usize;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        let scale = Tensor::from(1.0f32 / (self.head_dim as f32).sqrt()).cast(DType::F32);
        // The kv cache is `f16` while the stream is `f32`: each dot converts
        // its activation to the cache's type and leaves an `f32` result, which
        // is `ggml_mul_mat`'s contract again.
        let attn = q.cast(k.dtype()).matmul(k.transpose(2, 3)?)?.cast(DType::F32) * scale;

        let seq_len_i = seq_len.item::<i64>();
        let attn = if seq_len_i <= 1 {
            attn
        } else {
            let start = start_pos.item::<i64>();
            let mask = Tensor::full([seq_len_i, start + seq_len_i], f32::NEG_INFINITY).triu(start + 1)?.cast(attn.dtype());
            attn + mask
        };
        let out = attn.softmax([-1])?.cast(v.dtype()).matmul(&v)?.cast(DType::F32).transpose(1, 2)?;
        let d = self.num_heads * self.head_dim;
        let out = out.reshape([b_size, seq_len, d.into()])?;
        let xs = self.attn_output.forward(out)? + residual;

        let residual = xs.clone();
        let normed = self.ffn_norm.forward(&xs)?;
        // Materialize the projection before the activation. Fused with `swish`,
        // the matmul comes out as one thread per output doing a 3072-long
        // serial k-loop (4.6 ms at decode) plus two extra f32 scratch buffers
        // for the sigmoid chain, and the reduce-split candidates never apply;
        // the same matmul on its own is 176 us.
        let gate = self.ffn_gate.forward(normed.clone())?.contiguous()?.swish();
        let up = self.ffn_up.forward(normed)?;
        Ok(self.ffn_down.forward(gate * up)? + residual)
    }
}
