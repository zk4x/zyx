use crate::{LayerNorm, Linear, MultiheadAttention};
use zyx::{DType, Tensor, ZyxError};

/// A single layer of a Transformer decoder.
///
/// This layer implements the standard Transformer decoder operations:
/// 1. **Self-attention** on the target sequence.
/// 2. **Cross-attention** using the encoder output (memory).
/// 3. **Feedforward network** with activation function.
/// 4. **Residual connections** and **Layer Normalization**.
///
/// The behavior of the layer can be adjusted using `norm_first` (pre-norm vs post-norm),
/// dropout rate, and activation function.
pub struct TransformerDecoderLayer {
    self_attention: MultiheadAttention,
    cross_attention: MultiheadAttention,
    feedforward: Linear,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    dropout_rate: f32,                // Dropout rate passed as a parameter
    norm_first: bool,                 // Whether to apply norm before layers or after
    activation: fn(Tensor) -> Tensor, // Activation function
}

impl TransformerDecoderLayer {
    /// Creates a new `TransformerDecoderLayer`.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimensionality of input embeddings (number of features per token).
    /// * `nhead` - Number of attention heads in self-attention and cross-attention.
    /// * `dim_feedforward` - Hidden dimension of the feedforward network.
    /// * `dropout` - Dropout probability applied after attention and feedforward layers.
    /// * `activation` - Activation function applied after the feedforward network (e.g., ReLU).
    /// * `layer_norm_eps` - Small epsilon value for numerical stability in layer normalization.
    /// * `batch_first` - If true, input tensors have shape `[batch, seq, feature]`. Otherwise `[seq, batch, feature]`.
    /// * `norm_first` - Whether to apply layer normalization before sub-layers (pre-norm) or after (post-norm).
    /// * `bias` - Whether to include bias terms in linear and attention layers.
    /// * `dtype` - Data type of tensors (e.g., `DType::F32`, `DType::F64`).
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `TransformerDecoderLayer` or a `ZyxError` if initialization fails.
    pub fn new(
        d_model: usize,                   // embed_dim
        nhead: usize,                     // num_heads
        dim_feedforward: usize,           // dim_feedforward
        dropout: f32,                     // dropout rate
        activation: fn(Tensor) -> Tensor, // activation function
        layer_norm_eps: f64,              // layer_norm_eps
        batch_first: bool,                // batch_first
        norm_first: bool,                 // norm_first
        bias: bool,                       // use biases in layers
        dtype: DType,                     // tensor data type (e.g., f32, f64)
    ) -> Result<Self, ZyxError> {
        // Create self-attention and cross-attention layers
        let self_attention = MultiheadAttention::new(
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

        let cross_attention = MultiheadAttention::new(
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

        // Create feedforward layers (two Linear layers with ReLU activation)
        let feedforward = Linear::new(d_model, dim_feedforward, bias, dtype)?;

        // LayerNorms (First for attention, second for feedforward)
        let layer_norm_1 = LayerNorm::new(d_model, layer_norm_eps, true, bias, dtype)?;
        let layer_norm_2 = LayerNorm::new(d_model, layer_norm_eps, true, bias, dtype)?;

        // Return the complete TransformerDecoderLayer
        Ok(TransformerDecoderLayer {
            self_attention,
            cross_attention,
            feedforward,
            layer_norm_1,
            layer_norm_2,
            dropout_rate: dropout,
            norm_first,
            activation,
        })
    }

    /// Performs a forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `tgt` - Target sequence tensor (decoder input).
    /// * `memory` - Memory tensor from the encoder (encoder output).
    /// * `tgt_mask` - Optional mask for self-attention on the target sequence.
    /// * `memory_mask` - Optional mask for cross-attention on the memory sequence.
    /// * `tgt_key_padding_mask` - Optional padding mask for target tokens.
    /// * `memory_key_padding_mask` - Optional padding mask for memory tokens.
    /// * `tgt_is_causal` - Whether to apply causal masking to target self-attention (autoregressive decoding).
    /// * `memory_is_causal` - Whether to apply causal masking in cross-attention.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the output tensor of the decoder layer or a `ZyxError`.
    ///
    /// # Behavior
    ///
    /// 1. Applies layer normalization if `norm_first` is true.
    /// 2. Applies self-attention on the target sequence.
    /// 3. Applies residual connection and dropout.
    /// 4. Applies cross-attention with the encoder memory.
    /// 5. Applies residual connection and dropout.
    /// 6. Passes through feedforward network with activation.
    /// 7. Applies final residual connection and layer normalization.
    pub fn forward(
        &self,
        tgt: &Tensor,                           // Target sequence (input to the decoder)
        memory: &Tensor,                        // Memory sequence (encoder output)
        tgt_mask: Option<impl Into<Tensor>>, // Optional mask for target sequence (self-attention)
        memory_mask: Option<impl Into<Tensor>>, // Optional mask for memory sequence (cross-attention)
        tgt_key_padding_mask: Option<impl Into<Tensor>>, // Optional padding mask for target
        memory_key_padding_mask: Option<impl Into<Tensor>>, // Optional padding mask for memory
        tgt_is_causal: bool, // Whether to apply causal masking to target (autoregressive)
        memory_is_causal: bool, // Whether to apply causal masking to memory
    ) -> Result<Tensor, ZyxError> {
        let mut output = tgt.clone();

        // Apply LayerNorm first if norm_first is true
        if self.norm_first {
            output = self.layer_norm_1.forward(&output)?;
        }

        // Self-Attention: Apply self-attention to the target sequence
        let (attn_output, _) = self.self_attention.forward(
            &output,              // query = tgt
            &output,              // key = tgt
            &output,              // value = tgt
            tgt_key_padding_mask, // Padding mask for tgt
            true,                 // need_weights = true (return attention weights)
            tgt_mask,             // tgt_mask (optional)
            true,                 // average_attn_weights = true
            tgt_is_causal,        // Is causal self-attention (autoregressive)?
        )?;

        // Apply dropout after self-attention
        let attn_output = attn_output.dropout(self.dropout_rate);

        // Add residual connection after self-attention
        output = output + attn_output;

        // Apply LayerNorm after self-attention if norm_first is false
        if !self.norm_first {
            output = self.layer_norm_1.forward(&output)?;
        }

        // Cross-Attention: Apply cross-attention using memory (encoder output)
        let (cross_attn_output, _) = self.cross_attention.forward(
            &output,                 // query = tgt (output from self-attention)
            memory,                  // key = memory (encoder output)
            memory,                  // value = memory
            memory_key_padding_mask, // Padding mask for memory
            true,                    // need_weights = true (return attention weights)
            memory_mask,             // memory_mask (optional)
            true,                    // average_attn_weights = true
            memory_is_causal,        // Is causal attention for memory
        )?;

        // Apply dropout after cross-attention
        let cross_attn_output = cross_attn_output.dropout(self.dropout_rate);

        // Add residual connection after cross-attention
        output = output + cross_attn_output;

        // Apply LayerNorm after cross-attention if norm_first is false
        if !self.norm_first {
            output = self.layer_norm_2.forward(&output)?;
        }

        // Feedforward Network: Apply the feedforward layer
        let ff_output = self.feedforward.forward(&output)?;

        // Apply the activation function to the feedforward output
        let ff_output = (self.activation)(ff_output);

        // Apply dropout after feedforward
        let ff_output = ff_output.dropout(self.dropout_rate);

        // Add residual connection after feedforward
        output = output + ff_output;

        // Apply final LayerNorm after feedforward
        output = self.layer_norm_2.forward(&output)?;

        Ok(output)
    }
}
