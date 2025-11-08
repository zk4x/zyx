use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

use crate::{Activation, LayerNorm, Linear, MultiheadAttention};

/// TransformerEncoderLayer as in PyTorch's [`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
///
/// Applies multi-head self-attention followed by a feedforward network with residual connections and layer normalization.
/// Supports `batch_first` layout and `norm_first` pre/post normalization.
#[derive(Debug, Module)]
pub struct TransformerEncoderLayer {
    /// The number of expected features in the input (`d_model` in PyTorch)
    pub d_model: usize,
    /// The number of attention heads (`nhead`)
    pub nhead: usize,
    /// The dimension of the feedforward network (`dim_feedforward`)
    pub dim_feedforward: usize,
    /// Dropout probability applied after attention and feedforward layers
    pub dropout: f32,
    /// Small epsilon value for layer normalization to avoid division by zero
    pub layer_norm_eps: f64,
    /// If true, input/output tensors have shape `[batch, seq, features]`, otherwise `[seq, batch, features]`
    pub batch_first: bool,
    /// If true, layer normalization is applied before attention/feedforward ("pre-norm"), otherwise after ("post-norm")
    pub norm_first: bool,
    /// Whether linear layers include bias terms
    pub bias: bool,
    /// Multi-head self-attention module
    pub self_attn: MultiheadAttention,
    /// First linear layer in the feedforward network
    pub linear1: Linear,
    /// Second linear layer in the feedforward network
    pub linear2: Linear,
    /// Layer normalization after attention (or before if `norm_first = true`)
    pub norm1: LayerNorm,
    /// Layer normalization after feedforward (or before if `norm_first = true`)
    pub norm2: LayerNorm,
    /// Activation function for the feedforward network
    pub activation: Activation,
}

impl TransformerEncoderLayer {
    /// Creates a new `TransformerEncoderLayer`.
    ///
    /// # Arguments
    /// - `d_model`: Embedding dimension of input and output tensors.
    /// - `nhead`: Number of attention heads.
    /// - `dim_feedforward`: Dimension of the feedforward network.
    /// - `dropout`: Dropout probability applied after attention and feedforward layers.
    /// - `activation`: Activation function used in the feedforward network.
    /// - `layer_norm_eps`: Epsilon value for layer normalization.
    /// - `batch_first`: If true, input/output shape is `[batch, seq, features]`.
    /// - `norm_first`: If true, layer normalization is applied before sublayers.
    /// - `bias`: If true, linear layers have bias terms.
    /// - `dtype`: Data type of module parameters.
    ///
    /// # Returns
    /// Returns a configured `TransformerEncoderLayer` or an error if initialization fails.
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
        let self_attn = MultiheadAttention::new(
            d_model,
            nhead,
            dropout,
            bias,
            false, // add_bias_kv
            false, // add_zero_attn
            None,
            None,
            batch_first,
            dtype,
        )?;

        let linear1 = Linear::new(d_model, dim_feedforward, bias, dtype)?;
        let linear2 = Linear::new(dim_feedforward, d_model, bias, dtype)?;
        let norm1 = LayerNorm::new(d_model, bias, dtype)?;
        let norm2 = LayerNorm::new(d_model, bias, dtype)?;

        Ok(Self {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            activation,
        })
    }

    /// Forward pass through the `TransformerEncoderLayer`.
    ///
    /// # Arguments
    /// - `src`: Input tensor of shape `[B, T, E]` if `batch_first = true`, else `[T, B, E]`.
    /// - `src_mask`: Optional attention mask of shape `[T, T]`.
    /// - `src_key_padding_mask`: Optional padding mask of shape `[B, T]`.
    ///
    /// # Returns
    /// - Tensor of same shape as `src` (`[B, T, E]` or `[T, B, E]` depending on `batch_first`).
    ///
    /// # Notes
    /// - Dropout is applied after attention and feedforward sublayers.
    /// - Residual connections are applied in the standard way.
    /// - LayerNorm placement depends on `norm_first`.
    pub fn forward(
        &self,
        src: impl Into<Tensor>,
        src_mask: Option<Tensor>,
        src_key_padding_mask: Option<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let mut src = src.into();
        let residual = src.clone();
        let dropout = self.dropout;

        if self.norm_first {
            // Pre-norm
            let mut src2 = self
                .self_attn
                .forward(
                    src.clone(),
                    src.clone(),
                    src.clone(),
                    src_mask,
                    src_key_padding_mask,
                )?
                .0;
            src2 = src2.dropout(dropout);
            src = residual + src2;
            src = self.norm1.forward(src)?;

            let residual2 = src.clone();
            let mut src2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(&src)?)?)?;
            src2 = src2.dropout(dropout);
            src = residual2 + src2;
            src = self.norm2.forward(src)?;
        } else {
            // Post-norm (PyTorch default)
            let mut src2 = self
                .self_attn
                .forward(
                    src.clone(),
                    src.clone(),
                    src.clone(),
                    src_mask,
                    src_key_padding_mask,
                )?
                .0;
            src2 = src2.dropout(dropout);
            src = src + src2;
            src = self.norm1.forward(src)?;

            let mut src2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(&src)?))?;
            src2 = src2.dropout(dropout);
            src = src + src2;
            src = self.norm2.forward(src)?;
        }

        Ok(src)
    }
}
