// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Grouped-query causal attention with rotary position embeddings and a
//! preallocated kv-cache.
//!
//! The block owns its cache: layouts, offsets and the "one kernel shape for
//! every decode step" contract all belong together. Offsets are passed as
//! symbolic tensors (`Tensor::variable`), so a generation loop compiles once
//! regardless of how far `start_pos` advances.

use crate::Linear;
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Llama-3 style RoPE frequency scaling: a piecewise rescale of the
/// low-frequency end of the table.
#[derive(Debug, Clone, Copy)]
pub struct RopeScaling {
    /// Scale applied beyond the original context.
    pub factor: f64,
    /// Upper wavelength bound of the interpolated band.
    pub high_freq_factor: f64,
    /// Lower wavelength bound of the interpolated band.
    pub low_freq_factor: f64,
    /// Context length the unscaled table was built for.
    pub original_max_position_embeddings: i64,
}

/// Rotary position embedding parameters.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    /// Per-head width.
    pub head_dim: i64,
    /// Number of rows in the frequency table.
    pub max_position_embeddings: i64,
    /// RoPE base frequency.
    pub theta: f64,
    /// Optional long-context scaling.
    pub scaling: Option<RopeScaling>,
}

impl RopeConfig {
    /// Cosine and sine tables, each `[max_position_embeddings, head_dim / 2]`.
    fn tables(&self, dtype: DType) -> (Tensor, Tensor) {
        let head_dim = self.head_dim;
        let mut inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / self.theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        if let Some(scaling) = &self.scaling {
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
        let max_pos = self.max_position_embeddings;
        let t = Tensor::arange(0u32, max_pos as u32, 1)
            .unwrap()
            .cast(DType::F32)
            .reshape([max_pos, 1])
            .unwrap();
        let freqs = t.matmul(&inv_freq).unwrap();
        (freqs.cos().cast(dtype), freqs.sin().cast(dtype))
    }
}

fn causal_mask(size: i64) -> Tensor {
    let size_u = size as usize;
    let mask: Vec<f32> = (0..size_u)
        .flat_map(|i| (0..size_u).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from(mask).reshape([size, size]).unwrap()
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor, ZyxError> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let [b_sz, n_kv_head, seq_len, head_dim] = xs.dims::<4>()?;
    // `seq_len` may be symbolic: pass its dim tensor directly (NOT -1) so the
    // inferred dim keeps the original TensorId — merge-time provability checks
    // require the SAME dim tensor, not merely equal values.
    xs.unsqueeze(2)?
        .expand([-1i64, -1, n_rep as i64, -1, -1])?
        .reshape([b_sz, n_kv_head * (n_rep as i64), seq_len, head_dim])
}

fn rotate(xs: &Tensor, cos: &Tensor, sin: &Tensor, offset: &Tensor) -> Result<Tensor, ZyxError> {
    let [seq_len, _head_dim] = xs.rdims::<2>()?;
    let c = cos.narrow(0, offset.clone(), seq_len.clone())?;
    let s = sin.narrow(0, offset, seq_len)?;
    xs.rope(c, s)
}

/// Causal multi-head attention with grouped key/value heads, rotary
/// embeddings, and an owned kv-cache.
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Key cache, `[max_context, n_kv_head, head_dim]`. Owned state: written
    /// in place by every forward pass, so it must be realized alongside the
    /// outputs.
    pub cache_k: Tensor,
    /// Value cache, same shape as [`Self::cache_k`].
    pub cache_v: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl GqaAttention {
    /// Builds the block with random projections and a zeroed cache.
    ///
    /// # Errors
    ///
    /// Returns a shape error if any head/dimension combination is invalid.
    pub fn new(
        n_embd: i64,
        n_head: i64,
        n_kv_head: i64,
        rope: &RopeConfig,
        max_context: i64,
        dtype: DType,
    ) -> Result<GqaAttention, ZyxError> {
        if n_head % n_kv_head != 0 {
            return Err(ZyxError::shape_error(
                format!("n_head {n_head} is not a multiple of n_kv_head {n_kv_head}").into(),
            ));
        }
        let head_dim = rope.head_dim;
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;
        let (cos, sin) = rope.tables(dtype);
        Ok(GqaAttention {
            q_proj: Linear::new(n_embd, q_dim, false, dtype)?,
            k_proj: Linear::new(n_embd, kv_dim, false, dtype)?,
            v_proj: Linear::new(n_embd, kv_dim, false, dtype)?,
            o_proj: Linear::new(q_dim, n_embd, false, dtype)?,
            cache_k: Tensor::zeros([max_context, n_kv_head, head_dim], dtype).contiguous()?,
            cache_v: Tensor::zeros([max_context, n_kv_head, head_dim], dtype).contiguous()?,
            cos,
            sin,
            num_heads: n_head,
            num_kv_heads: n_kv_head,
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
        let [b_size, seq_len, _n_embd] = xs.dims::<3>()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape([b_size.clone(), seq_len.clone(), self.num_heads.into(), self.head_dim.into()])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b_size.clone(), seq_len.clone(), self.num_kv_heads.into(), self.head_dim.into()])?
            .transpose(1, 2)?;
        let v = v
            .reshape([b_size.clone(), seq_len.clone(), self.num_kv_heads.into(), self.head_dim.into()])?
            .transpose(1, 2)?;

        let q = rotate(&q, &self.cos, &self.sin, start_pos)?;
        let k = rotate(&k, &self.cos, &self.sin, start_pos)?;

        let k_new = k.squeeze([0]).transpose(0, 1)?;
        let v_new = v.squeeze([0]).transpose(0, 1)?;
        self.cache_k.narrow(0, start_pos, &seq_len)?.assign(&k_new)?;
        self.cache_v.narrow(0, start_pos, &seq_len)?.assign(&v_new)?;

        let k = self.cache_k.narrow(0, 0i64, cache_len.clone())?.unsqueeze(0)?.transpose(1, 2)?;
        let v = self.cache_v.narrow(0, 0i64, cache_len.clone())?.unsqueeze(0)?.transpose(1, 2)?;

        let n_rep = (self.num_heads / self.num_kv_heads) as usize;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        let scale = Tensor::from(1.0f32 / (self.head_dim as f32).sqrt()).cast(q.dtype());
        let attn = q.matmul(k.transpose(2, 3)?)? * scale;

        let seq_len_i = seq_len.item::<i64>();
        let attn = if seq_len_i <= 1 {
            attn
        } else {
            let mask = causal_mask(seq_len_i).cast(attn.dtype());
            attn + mask
        };
        let out = attn.softmax([-1])?.matmul(&v)?.transpose(1, 2)?;
        let d = self.num_heads * self.head_dim;
        let out = out.reshape([b_size, seq_len, d.into()])?;
        self.o_proj.forward(out)
    }
}
