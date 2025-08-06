use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

use crate::Linear;

/// Implements multi-head attention as described in "Attention Is All You Need".
///
/// This module is fully compatible with PyTorch’s `torch.nn.MultiheadAttention`,
/// supporting all core features including key/value projections, bias vectors,
/// dropout, and optional shape/batch layout controls.
#[derive(Debug, Module)]
pub struct MultiheadAttention {
    /// - `embed_dim`: Total dimension of the model (i.e. output embedding size).
    pub embed_dim: usize,
    /// - `kdim`: Dimension of the key input. If `None`, defaults to `embed_dim`.
    pub kdim: usize,
    /// - `vdim`: Dimension of the value input. If `None`, defaults to `embed_dim`.
    pub vdim: usize,
    /// - `num_heads`: Number of parallel attention heads.
    pub num_heads: usize,
    /// - `head_dim`: Dimension per attention head (i.e. `embed_dim / num_heads`).
    pub head_dim: usize,

    /// - `q_proj`: Linear projection layer for the query.
    pub q_proj: Linear,
    /// - `k_proj`: Linear projection layer for the key.
    pub k_proj: Linear,
    /// - `v_proj`: Linear projection layer for the value.
    pub v_proj: Linear,
    /// - `out_proj`: Final linear projection layer for the output.
    pub out_proj: Linear,

    /// - `dropout`: Dropout probability applied to attention weights.
    pub dropout: f32,
    /// - `add_bias_kv`: If true, learned bias vectors are added to key and value.
    pub add_bias_kv: bool,
    /// - `add_zero_attn`: If true, zero vectors are appended to key and value sequences.
    pub add_zero_attn: bool,
    /// - `batch_first`: If true, input and output tensors use shape `[B, T, E]`; otherwise `[T, B, E]`.
    pub batch_first: bool,

    /// - `bias_k`: Optional learnable bias added to key (shape `[1, 1, embed_dim]`).
    pub bias_k: Option<Tensor>,
    /// - `bias_v`: Optional learnable bias added to value (shape `[1, 1, embed_dim]`).
    pub bias_v: Option<Tensor>,
}

impl MultiheadAttention {
    /// Creates a PyTorch-compatible MultiheadAttention module.
    ///
    /// # Arguments
    /// - `embed_dim`: Total embedding dimension.
    /// - `num_heads`: Number of attention heads.
    /// - `dropout`: Dropout probability on attention weights.
    /// - `bias`: Whether to include bias terms in projections.
    /// - `add_bias_kv`: If true, adds learned bias to key and value.
    /// - `add_zero_attn`: If true, appends zero vector to key and value sequences.
    /// - `kdim`: Optional key dimension. Defaults to `embed_dim`.
    /// - `vdim`: Optional value dimension. Defaults to `embed_dim`.
    /// - `batch_first`: If true, expects input shape `[B, T, E]`. Else `[T, B, E]`.
    /// - `dtype`: DType of internal parameters and tensors.
    ///
    /// # Returns
    /// A configured `MultiheadAttention` module, or error on shape issues.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: f32,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
        batch_first: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        if embed_dim % num_heads != 0 {
            return Err(ZyxError::ShapeError(
                format!(
                    "embed_dim ({}) must be divisible by num_heads ({})",
                    embed_dim, num_heads
                )
                .into(),
            ));
        }

        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);
        let head_dim = embed_dim / num_heads;

        let q_proj = Linear::new(embed_dim, embed_dim, bias, dtype)?;
        let k_proj = Linear::new(kdim, embed_dim, bias, dtype)?;
        let v_proj = Linear::new(vdim, embed_dim, bias, dtype)?;
        let out_proj = Linear::new(embed_dim, embed_dim, bias, dtype)?;

        let (bias_k, bias_v) = if add_bias_kv {
            (
                Some(Tensor::zeros([1, 1, embed_dim], dtype)),
                Some(Tensor::zeros([1, 1, embed_dim], dtype)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            embed_dim,
            kdim,
            vdim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            dropout,
            add_bias_kv,
            add_zero_attn,
            batch_first,
            bias_k,
            bias_v,
        })
    }

    /// Forward multihead attention
    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Option<Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor), ZyxError> {
        let (mut q, mut k, mut v) = (query, key, value);

        if !self.batch_first {
            q = q.transpose(0, 1)?;
            k = k.transpose(0, 1)?;
            v = v.transpose(0, 1)?;
        }

        let [b, t_q, _] = q.dims::<3>()?;
        let [_, t_kv, _] = k.dims::<3>()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self
            .q_proj
            .forward(q)?
            .reshape([b, t_q, h, d])?
            .transpose(1, 2)?; // [B, H, T_q, D]
        let mut k = self
            .k_proj
            .forward(k)?
            .reshape([b, t_kv, h, d])?
            .transpose(1, 2)?; // [B, H, T_kv, D]
        let mut v = self
            .v_proj
            .forward(v)?
            .reshape([b, t_kv, h, d])?
            .transpose(1, 2)?; // [B, H, T_kv, D]

        if self.add_bias_kv {
            if let (Some(bk), Some(bv)) = (&self.bias_k, &self.bias_v) {
                let bk = bk
                    .expand([b, 1, self.embed_dim])?
                    .reshape([b, 1, h, d])?
                    .transpose(1, 2)?; // [B, H, 1, D]
                let bv = bv
                    .expand([b, 1, self.embed_dim])?
                    .reshape([b, 1, h, d])?
                    .transpose(1, 2)?;
                k = Tensor::cat([&k, &bk], 2)?;
                v = Tensor::cat([&v, &bv], 2)?;
            }
        }

        if self.add_zero_attn {
            let zero = Tensor::zeros([b, h, 1, d], k.dtype());
            k = Tensor::cat([&k, &zero], 2)?;
            v = Tensor::cat([&v, &zero], 2)?;
        }

        let scale = (d as f32).sqrt();
        let mut attn_scores = q.matmul(k.transpose(-2, -1)?)? / scale; // [B, H, T_q, T_kv]

        if let Some(mask) = attn_mask {
            attn_scores = attn_scores + mask;
        }

        let attn_weights = attn_scores.softmax([-1])?; // [B, H, T_q, T_kv]
        let attn_weights = if train {
            attn_weights.dropout(self.dropout)
        } else {
            attn_weights
        };

        let attn_output = attn_weights.matmul(v)?; // [B, H, T_q, D]
        let attn_output = attn_output.transpose(1, 2)?.reshape([b, t_q, h * d])?; // [B, T_q, E]

        let output = self.out_proj.forward(attn_output)?;

        let output = if self.batch_first {
            output
        } else {
            output.transpose(0, 1)? // [T_q, B, E]
        };

        Ok((output, attn_weights))
    }
}
