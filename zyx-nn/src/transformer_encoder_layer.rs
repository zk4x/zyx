use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

use crate::{Activation, LayerNorm, MultiheadAttention};

/// TransformerEncoderLayer as in PyTorch's torch.nn.TransformerEncoderLayer
#[derive(Debug, Module)]
pub struct TransformerEncoderLayer {
    /// the embedding dimension (d_model)
    pub d_model: usize,
    /// the number of attention heads (nhead)
    pub nhead: usize,
    /// dimension of the feedforward network (dim_feedforward)
    pub dim_feedforward: usize,
    /// dropout probability
    pub dropout: f32,
    /// epsilon for layer norm
    pub layer_norm_eps: f64,
    /// use batch_first layout ([B, T, E]) if true, else ([T, B, E])
    pub batch_first: bool,
    /// if true, apply layer norm before attention and feedforward ("pre-norm")
    pub norm_first: bool,
    /// whether linear layers have bias
    pub bias: bool,
    /// attention module
    pub self_attn: MultiheadAttention,
    /// first linear in feedforward
    pub linear1: crate::Linear,
    /// second linear in feedforward
    pub linear2: crate::Linear,
    /// layer norm after attention or before (depending on norm_first)
    pub norm1: LayerNorm,
    /// layer norm after feedforward or before (depending on norm_first)
    pub norm2: LayerNorm,
    /// activation function in feedforward
    pub activation: Activation,
}

impl TransformerEncoderLayer {
    /// Creates a new TransformerEncoderLayer module.
    ///
    /// # Arguments
    /// - `d_model`: The number of expected features in the input (embedding dimension).
    /// - `nhead`: The number of attention heads.
    /// - `dim_feedforward`: The dimension of the feedforward network model (default 2048).
    /// - `dropout`: Dropout probability applied after attention and feedforward layers (default 0.1).
    /// - `activation`: The activation function to use in the feedforward layer (default ReLU).
    /// - `layer_norm_eps`: A small epsilon value for layer normalization to avoid division by zero (default 1e-5).
    /// - `batch_first`: If true, input and output tensors are expected with shape `[batch, seq, features]`
    ///                  otherwise `[seq, batch, features]` (default false).
    /// - `norm_first`: If true, layer normalization is done before each sublayer, otherwise after (default false).
    /// - `bias`: If true, linear layers include bias terms (default true).
    /// - `dtype`: The data type of the module parameters.
    ///
    /// # Returns
    /// Returns a configured TransformerEncoderLayer instance or an error if initialization fails (e.g., shape mismatch).
    #[allow(clippy::too_many_arguments)]
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
        // Create the multi-head attention module
        let self_attn = MultiheadAttention::new(
            d_model,
            nhead,
            dropout,
            bias,
            false, // add_bias_kv false for encoder
            false, // add_zero_attn false for encoder
            None,
            None,
            batch_first,
            dtype,
        )?;

        // Create the feedforward linear layers
        let linear1 = crate::Linear::new(d_model, dim_feedforward, bias, dtype)?;
        let linear2 = crate::Linear::new(dim_feedforward, d_model, bias, dtype)?;

        // Layer norms
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

    /// Forward pass through TransformerEncoderLayer.
    ///
    /// # Arguments
    /// - `src`: input tensor ([B, T, E] if batch_first else [T, B, E])
    ///
    /// # Returns
    /// Output tensor of the same shape as input.
    pub fn forward(&self, src: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let mut src = src.into();

        let dropout = self.dropout;

        let residual = src.clone();

        if self.norm_first {
            // Pre-norm
            let src2 = self
                .self_attn
                .forward(src.clone(), src.clone(), src.clone(), None::<Tensor>)?
                .0;
            let src2 = src2.dropout(dropout);
            src = residual + src2;
            let src = self.norm1.forward(src)?;

            let residual = src.clone();
            let mut src2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(src.clone())?))?;
            src2 = src2.dropout(dropout);
            let src = residual + src2;
            let src = self.norm2.forward(src)?;
            Ok(src)
        } else {
            // Post-norm (default in PyTorch)
            let src2 = self
                .self_attn
                .forward(src.clone(), src.clone(), src.clone(), None::<Tensor>)?
                .0;
            let src2 = src2.dropout(dropout);
            src = src + src2;
            src = self.norm1.forward(src)?;

            let mut src2 = self
                .linear2
                .forward(self.activation.forward(&self.linear1.forward(&src)?))?;
            src2 = src2.dropout(dropout);
            src = src + src2;
            src = self.norm2.forward(src)?;

            Ok(src)
        }
    }
}
