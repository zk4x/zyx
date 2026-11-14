// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Hyperparameters of the `qwen35` trunk, read from the GGUF metadata.
//!
//! Nothing here is hardcoded to the 27B checkpoint: every dimension comes
//! out of the file, so a different qwen35 export loads with the same code.
//! The only derived values are the ones the architecture defines
//! (`key_dim = state_size * group_count`, `conv_dim = 2*key_dim + inner`).

use zyx::{GGUFMetadata, ZyxError};

/// Trunk geometry of the qwen35 checkpoint.
#[derive(Debug, Clone)]
pub struct Config {
    /// Decoder blocks in the trunk (MTP layers excluded).
    pub n_blocks: i64,
    /// Residual stream width.
    pub hidden: i64,
    /// SwiGLU inner width.
    pub ffn: i64,
    /// Full-attention heads.
    pub n_head: i64,
    /// Full-attention key/value heads.
    pub n_kv: i64,
    /// Full-attention head width.
    pub head_dim: i64,
    /// Rotated dimensions per head (partial RoPE).
    pub rot_dim: i64,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub eps: f64,
    /// One full-attention block every `full_attention_interval` blocks.
    pub full_attention_interval: i64,
    /// Width of one linear-attention key head.
    pub ssm_state: i64,
    /// Key heads of the linear-attention block.
    pub ssm_k_heads: i64,
    /// Value heads of the linear-attention block.
    pub ssm_v_heads: i64,
    /// Width of one linear-attention value head.
    pub ssm_v_head_dim: i64,
    /// Depthwise conv kernel width.
    pub conv_kernel: i64,
}

impl Config {
    /// Reads the qwen35 hyperparameters out of a GGUF metadata map.
    pub fn from_metadata(metadata: &GGUFMetadata) -> Result<Self, ZyxError> {
        let nextn = metadata.opt_int("qwen35.nextn_predict_layers")?.unwrap_or(0);
        if nextn > 0 {
            return Err(ZyxError::parse_error(
                format!("qwen35.nextn_predict_layers = {nextn}: this loader only implements the non-MTP trunk").into(),
            ));
        }
        let ssm_state = metadata.int("qwen35.ssm.state_size")?;
        let ssm_k_heads = metadata.int("qwen35.ssm.group_count")?;
        let ssm_v_heads = metadata.int("qwen35.ssm.time_step_rank")?;
        let ssm_inner = metadata.int("qwen35.ssm.inner_size")?;
        if ssm_inner % ssm_v_heads != 0 {
            return Err(ZyxError::parse_error(
                format!("qwen35.ssm.inner_size {ssm_inner} is not divisible by time_step_rank {ssm_v_heads}").into(),
            ));
        }
        Ok(Self {
            n_blocks: metadata.int("qwen35.block_count")?,
            hidden: metadata.int("qwen35.embedding_length")?,
            ffn: metadata.int("qwen35.feed_forward_length")?,
            n_head: metadata.int("qwen35.attention.head_count")?,
            n_kv: metadata.int("qwen35.attention.head_count_kv")?,
            head_dim: metadata.int("qwen35.attention.key_length")?,
            rot_dim: metadata.int("qwen35.rope.dimension_count")?,
            rope_theta: metadata.float("qwen35.rope.freq_base")?,
            eps: metadata.float("qwen35.attention.layer_norm_rms_epsilon")?,
            full_attention_interval: metadata.int("qwen35.full_attention_interval")?,
            ssm_state,
            ssm_k_heads,
            ssm_v_heads,
            ssm_v_head_dim: ssm_inner / ssm_v_heads,
            conv_kernel: metadata.int("qwen35.ssm.conv_kernel")?,
        })
    }

    /// Key width of one linear-attention block.
    #[must_use]
    pub fn ssm_key_dim(&self) -> i64 {
        self.ssm_state * self.ssm_k_heads
    }

    /// Value width of one linear-attention block.
    #[must_use]
    pub fn ssm_value_dim(&self) -> i64 {
        self.ssm_state * self.ssm_v_heads
    }

    /// Concatenated q/k/v width fed to the depthwise conv.
    #[must_use]
    pub fn conv_dim(&self) -> i64 {
        2 * self.ssm_key_dim() + self.ssm_value_dim()
    }

    /// `true` when block `i` is a full-attention block.
    #[must_use]
    pub fn is_full(&self, i: i64) -> bool {
        (i + 1) % self.full_attention_interval == 0
    }
}
