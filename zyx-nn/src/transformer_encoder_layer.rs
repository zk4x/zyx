use zyx::{Tensor, DType, ZyxError};
use crate::{MultiheadAttention, Linear, LayerNorm};

/// A single Transformer Encoder layer, analogous to `torch.nn.TransformerEncoderLayer`.
///
/// This layer implements a standard Transformer encoder block with a multi-head self-attention
/// mechanism followed by a position-wise feedforward network. Layer normalization can be applied
/// either before ("pre-norm") or after ("post-norm") the attention and feedforward sub-layers.
pub struct TransformerEncoderLayer {
    /// - `self_attn`: The multi-head self-attention module.
    pub self_attn: MultiheadAttention,
    /// - `linear1`: The first linear layer of the feedforward network (expansion).
    pub linear1: Linear,
    /// - `dropout`: Dropout probability applied after attention and feedforward layers.
    pub dropout: f32,
    /// - `linear2`: The second linear layer of the feedforward network (projection back to `d_model`).
    pub linear2: Linear,
    /// - `norm1`: LayerNorm applied after the self-attention block (or before if `norm_first` is true).
    pub norm1: LayerNorm,
    /// - `norm2`: LayerNorm applied after the feedforward block (or before if `norm_first` is true).
    pub norm2: LayerNorm,
    /// - `activation`: The activation function used in the feedforward network (e.g., ReLU, GELU).
    pub activation: fn(Tensor) -> Tensor,
    /// - `norm_first`: If `true`, applies layer normalization before each sub-layer (pre-norm).
    pub norm_first: bool,
    /// - `batch_first`: If `true`, expects input tensors of shape `(batch_size, seq_len, d_model)`.
    pub batch_first: bool,
}

impl TransformerEncoderLayer {
    /// Constructs a new `TransformerEncoderLayer`.
    ///
    /// # Arguments
    ///
    /// * `d_model` - The number of expected features in the input (embedding size).
    /// * `nhead` - The number of attention heads.
    /// * `dim_feedforward` - The dimension of the feedforward network.
    /// * `dropout` - Dropout probability applied after attention and feedforward layers.
    /// * `activation` - Activation function used in the feedforward network.
    /// * `layer_norm_eps` - Epsilon value for numerical stability in layer normalization.
    /// * `batch_first` - If `true`, input/output tensors are expected in `(batch, seq, feature)` format.
    /// * `norm_first` - If `true`, applies layer normalization before sub-layers (pre-norm).
    /// * `bias` - If `true`, linear layers include bias terms.
    /// * `dtype` - The data type of the layer’s parameters and outputs.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `TransformerEncoderLayer` or a `ZyxError`.
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout: f32,
        activation: fn(Tensor) -> Tensor,
        layer_norm_eps: f64,
        batch_first: bool,
        norm_first: bool,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, ZyxError> {
        // --- Multihead self-attention ---
        let self_attn = MultiheadAttention::new(
            d_model,
            nhead,
            dropout,
            bias,
            /* add_bias_kv */ false,
            /* add_zero_attn */ false,
            /* kdim */ None,
            /* vdim */ None,
            batch_first,
            dtype,
        )?;

        // --- Feedforward network ---
        let linear1 = Linear::new(d_model, dim_feedforward, bias, dtype)?;
        let linear2 = Linear::new(dim_feedforward, d_model, bias, dtype)?;

        // --- LayerNorms ---
        let norm1 = LayerNorm::new(d_model, layer_norm_eps, true, bias, dtype)?;
        let norm2 = LayerNorm::new(d_model, layer_norm_eps, true, bias, dtype)?;

        Ok(Self {
            self_attn,
            linear1,
            dropout,
            linear2,
            norm1,
            norm2,
            activation,
            norm_first,
            batch_first,
        })
    }

    /// Performs a forward pass of the Transformer encoder layer.
    ///
    /// # Arguments
    ///
    /// * `src` - Input tensor of shape `(seq_len, batch_size, d_model)` or `(batch_size, seq_len, d_model)` if `batch_first`.
    /// * `src_mask` - Optional attention mask tensor to prevent attention to certain positions.
    /// * `src_key_padding_mask` - Optional mask tensor for padding positions in the input.
    ///
    /// # Returns
    ///
    /// A `Result` containing the output tensor after applying self-attention and feedforward blocks.
    pub fn forward(
        &self,
        src: impl Into<Tensor>,
        src_mask: Option<Tensor>,
        src_key_padding_mask: Option<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let mut x = src.into();

        if self.norm_first {
            // Pre-norm variant
            let sa_out = self.self_attention_block(
                self.norm1.forward(&x)?,
                &src_mask,
                &src_key_padding_mask,
            )?;
            x = x + sa_out;

            let ff_out = self.feed_forward_block(self.norm2.forward(&x)?)?;
            x = x + ff_out;
        } else {
            // Post-norm variant
            let sa_out = self.self_attention_block(&x, &src_mask, &src_key_padding_mask)?;
            x = self.norm1.forward(x + sa_out)?;

            let ff_out = self.feed_forward_block(&x)?;
            x = self.norm2.forward(x + ff_out)?;
        }

        Ok(x)
    }

    fn self_attention_block(
        &self,
        x: impl Into<Tensor>,
        src_mask: &Option<Tensor>,
        src_key_padding_mask: &Option<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let (attn_output, _weights) = self.self_attn.forward(
            x.clone(),
            x.clone(), // key = query = value
            x,
            src_key_padding_mask.as_ref(),
            /* need_weights */ false,
            src_mask.as_ref(),
            /* average_attn_weights */ true,
            /* is_causal */ false,
        )?;
        // Dropout after attention output
        Ok(attn_output.dropout(self.dropout))
    }

    fn feed_forward_block(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let x = self.linear1.forward(&x)?;
        let x = (self.activation)(x);
        let x = x.dropout(self.dropout);
        let x = self.linear2.forward(&x)?;
        Ok(x.dropout(self.dropout))
    }
}
