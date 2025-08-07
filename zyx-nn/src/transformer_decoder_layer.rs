use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

use crate::{Activation, LayerNorm, Linear, MultiheadAttention};

/// Transformer Decoder Layer module.
///
/// Implements one layer of the Transformer decoder as described in "Attention Is All You Need".
///
/// This layer consists of:
/// - Self-attention with optional masking to prevent attending to future tokens.
/// - Multihead attention over encoder outputs (memory).
/// - Position-wise feedforward network.
/// - Layer normalization applied either before or after each sub-layer (configurable).
/// - Dropout applied after attention and feedforward layers.
///
/// Compatible with PyTorch's `torch.nn.TransformerDecoderLayer`.
#[derive(Debug, Module)]
pub struct TransformerDecoderLayer {
    self_attn: MultiheadAttention,
    multihead_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: f32,
    dropout1: f32,
    dropout2: f32,
    activation: Activation,
    norm_first: bool,
}

impl TransformerDecoderLayer {
    /// Creates a new `TransformerDecoderLayer`.
    ///
    /// # Arguments
    ///
    /// * `d_model` - the number of expected features in the input (embedding dimension).
    /// * `nhead` - the number of heads in the multiheadattention models.
    /// * `dim_feedforward` - the dimension of the feedforward network model (default 2048).
    /// * `dropout` - the dropout value (default 0.1).
    /// * `activation` - the activation function of intermediate layer, e.g., ReLU or GELU.
    /// * `layer_norm_eps` - epsilon value for layer normalization (default 1e-5).
    /// * `batch_first` - if true, input and output tensors are provided as (batch, seq, feature).
    /// * `norm_first` - if true, layer norm is applied before each sublayer (default false).
    /// * `bias` - if true, add bias to linear layers (default true).
    /// * `dtype` - data type for tensors.
    ///
    /// # Returns
    ///
    /// Returns `Result<Self, ZyxError>` which is the constructed decoder layer or error on invalid params.
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout: f32,
        activation: Activation,
        layer_norm_eps: f64,
        batch_first: bool,
        norm_first: bool,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        // Create self-attention and multi-head attention modules
        let self_attn = MultiheadAttention::new(
            d_model,
            nhead,
            dropout,
            bias,
            false, // add_bias_kv false for decoder self-attention
            false, // add_zero_attn false
            None,
            None,
            batch_first,
            dtype,
        )?;

        let multihead_attn = MultiheadAttention::new(
            d_model,
            nhead,
            dropout,
            bias,
            false,
            false,
            None,
            None,
            batch_first,
            dtype,
        )?;

        // Feedforward layers
        let linear1 = Linear::new(d_model, dim_feedforward, bias, dtype)?;
        let linear2 = Linear::new(dim_feedforward, d_model, bias, dtype)?;

        // Layer norms
        let mut norm1 = LayerNorm::new(d_model, bias, dtype)?;
        norm1.eps = layer_norm_eps;
        let mut norm2 = LayerNorm::new(d_model, bias, dtype)?;
        norm2.eps = layer_norm_eps;
        let mut norm3 = LayerNorm::new(d_model, bias, dtype)?;
        norm3.eps = layer_norm_eps;

        Ok(Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            dropout,
            dropout1: dropout,
            dropout2: dropout,
            activation,
            norm_first,
        })
    }

    /// Forward pass of the Transformer decoder layer.
    ///
    /// # Arguments
    ///
    /// * `tgt` - target sequence tensor of shape `[seq_len, batch, d_model]` or `[batch, seq_len, d_model]` if batch_first.
    /// * `memory` - encoder output tensor, same batch and feature dims.
    /// * `tgt_mask` - optional mask for target sequence (e.g. subsequent masking).
    /// * `memory_mask` - optional mask for memory sequence.
    /// * `tgt_key_padding_mask` - optional mask for target keys per batch.
    /// * `memory_key_padding_mask` - optional mask for memory keys per batch.
    /// * `train` - training mode (enables dropout).
    ///
    /// # Returns
    ///
    /// Output tensor and optionally attention weights can be returned if needed.
    pub fn forward(
        &self,
        tgt: impl Into<Tensor>,
        memory: impl Into<Tensor>,
        tgt_mask: Option<impl Into<Tensor>>,
        memory_mask: Option<impl Into<Tensor>>,
        tgt_key_padding_mask: Option<impl Into<Tensor>>,
        memory_key_padding_mask: Option<impl Into<Tensor>>,
        train: bool,
    ) -> Result<Tensor, ZyxError> {
        let mut tgt = tgt.into();
        let memory = memory.into();

        // Apply normalization before or after sublayers as configured
        if self.norm_first {
            tgt = self.norm1.forward(tgt.clone())?;
            let tgt2 = self
                .self_attn
                .forward(tgt.clone(), tgt.clone(), tgt.clone(), tgt_mask, train)?
                .0;
            tgt = tgt + tgt2.dropout(self.dropout1);

            tgt = self.norm2.forward(tgt.clone())?;
            let tgt2 = self
                .multihead_attn
                .forward(
                    tgt.clone(),
                    memory.clone(),
                    memory.clone(),
                    memory_mask,
                    train,
                )?
                .0;
            tgt = tgt + tgt2.dropout(self.dropout2);

            let tgt2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(tgt.clone())?))?
                .dropout(self.dropout);

            tgt = tgt + tgt2;
            tgt = self.norm3.forward(tgt)?;
        } else {
            // Post-norm version
            let tgt2 = self
                .self_attn
                .forward(tgt.clone(), tgt.clone(), tgt.clone(), tgt_mask, train)?
                .0;
            tgt = tgt + tgt2.dropout(self.dropout1);
            tgt = self.norm1.forward(tgt)?;

            let tgt2 = self
                .multihead_attn
                .forward(
                    tgt.clone(),
                    memory.clone(),
                    memory.clone(),
                    memory_mask,
                    train,
                )?
                .0;
            tgt = tgt + tgt2.dropout(self.dropout2);
            tgt = self.norm2.forward(tgt)?;

            let tgt2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(tgt.clone())?))?
                .dropout(self.dropout);
            tgt = tgt + tgt2;
            tgt = self.norm3.forward(tgt)?;
        }

        Ok(tgt)
    }
}
