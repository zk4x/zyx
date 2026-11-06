// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Pre-norm decoder block: attention and feed-forward each wrapped in a
//! residual with an RMSNorm on the branch input.

use crate::{GqaAttention, RMSNorm, SwiGlu};
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Pre-norm decoder block with grouped-query attention and a SwiGLU MLP.
#[derive(Debug, Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct DecoderBlock {
    /// attention
    pub self_attn: GqaAttention,
    /// feed-forward
    pub mlp: SwiGlu,
    /// pre-attention norm
    pub input_layernorm: RMSNorm,
    /// pre-MLP norm
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderBlock {
    /// Assembles a block from its parts.
    #[must_use]
    pub fn new(self_attn: GqaAttention, mlp: SwiGlu, n_embd: i64, eps: f64, dtype: DType) -> DecoderBlock {
        let mut input_layernorm = RMSNorm::new(n_embd, dtype);
        input_layernorm.eps = eps;
        let mut post_attention_layernorm = RMSNorm::new(n_embd, dtype);
        post_attention_layernorm.eps = eps;
        DecoderBlock {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass.
    ///
    /// `start_pos` and `cache_len` are forwarded to the attention cache.
    ///
    /// # Errors
    ///
    /// Returns a shape error if `xs` is not rank 3.
    pub fn forward(&mut self, xs: &Tensor, start_pos: &Tensor, cache_len: &Tensor) -> Result<Tensor, ZyxError> {
        let residual = xs.clone();
        let normed = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&normed, start_pos, cache_len)? + residual;
        let residual = xs.clone();
        let normed = self.post_attention_layernorm.forward(xs)?;
        Ok(self.mlp.forward(&normed)? + residual)
    }
}
